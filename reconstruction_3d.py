import numpy as np
import cv2
import trimesh
from typing import List, Tuple, Optional
import numpy.linalg as LA


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normaliza puntos 2D para mejorar la estabilidad numérica.
    """
    centroid = np.mean(points, axis=0)
    std_dev = np.std(points[:, 0] ** 2 + points[:, 1] ** 2)
    norm_factor = np.sqrt(2) / std_dev if std_dev > 0 else 1.0

    T = np.array([
        [norm_factor, 0, -norm_factor * centroid[0]],
        [0, norm_factor, -norm_factor * centroid[1]],
        [0, 0, 1]
    ])

    normalized_points = np.column_stack([points, np.ones(len(points))])
    normalized_points = (T @ normalized_points.T).T[:, :2]

    return normalized_points, T


def find_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz fundamental usando el algoritmo de 8 puntos normalizado.
    """
    # Normalizar puntos
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    # Construir matriz A para resolver Af = 0
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1 = norm_pts1[i]
        x2, y2 = norm_pts2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Resolver usando SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Forzar rango 2
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    # Desnormalizar
    F = T2.T @ F @ T1

    return F / F[2, 2]


def triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangula puntos 3D usando DLT (Direct Linear Transform) con validaciones adicionales.
    """
    if pts1.shape[0] != 2:
        pts1 = pts1.T
    if pts2.shape[0] != 2:
        pts2 = pts2.T

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    assert P1.shape == (3, 4), "P1 debe ser de tamaño (3, 4)"
    assert P2.shape == (3, 4), "P2 debe ser de tamaño (3, 4)"
    assert pts1.shape[1] == pts2.shape[1], "Los puntos no coinciden en cantidad"

    try:
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = (points_4d[:3] / points_4d[3]).T
        return points_3d
    except Exception as e:
        raise RuntimeError(f"Error en triangulación: {str(e)}")


def get_camera_poses(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extrae las posibles poses de cámara de la matriz esencial.
    """
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def check_pose(R: np.ndarray, t: np.ndarray, points_3d: np.ndarray) -> bool:
    """
    Verifica si una pose es válida (puntos delante de ambas cámaras).
    """
    P = np.hstack((R, t.reshape(-1, 1)))
    for X in points_3d:
        # Verificar profundidad en primera cámara
        X_homog = np.append(X, 1)
        x1 = np.dot([0, 0, 1], X_homog)
        # Verificar profundidad en segunda cámara
        x2 = np.dot([0, 0, 1], np.dot(P, X_homog))
        if x1 < 0 or x2 < 0:
            return False
    return True


def filter_points(points_3d: np.ndarray, max_dist: float = 100.0) -> np.ndarray:
    """
    Filtra puntos 3D basado en distancia y estadísticas.
    """
    # Eliminar puntos muy lejanos
    distances = LA.norm(points_3d, axis=1)
    mask_dist = distances < max_dist

    # Filtrar outliers usando estadísticas
    mean = np.mean(points_3d[mask_dist], axis=0)
    std = np.std(points_3d[mask_dist], axis=0)
    mask_stats = np.all(np.abs(points_3d - mean) < 3 * std, axis=1)

    return points_3d[mask_dist & mask_stats]


def reconstruct_3d(image_paths: List[str], output_path: str, progress_callback: Optional[callable] = None) -> None:
    """
    Reconstrucción 3D a partir de múltiples imágenes y creación directa de una malla.
    """
    images = [cv2.imread(path) for path in image_paths]  # Leer imágenes directamente
    total_steps = len(images) * 3 + 1  # Detección, emparejamiento, reconstrucción y malla

    # Detectar características con SIFT
    sift = cv2.SIFT_create()
    keypoints_list, descriptors_list = [], []

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        if progress_callback:
            progress_callback((idx + 1) / total_steps)

    # Emparejar características entre imágenes consecutivas
    bf = cv2.BFMatcher()
    all_matches = []

    for i in range(len(descriptors_list) - 1):
        matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Ratio test de Lowe
                good_matches.append(m)
        all_matches.append(good_matches)

        if progress_callback:
            progress_callback((len(keypoints_list) + i + 1) / total_steps)

    # Reconstrucción 3D iterativa
    reconstructed_points = []
    current_R = np.eye(3)  # Matriz de rotación inicial
    current_t = np.zeros((3, 1))  # Traslación inicial

    for i, matches in enumerate(all_matches):
        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        # Actualizamos la posición acumulada
        current_R = R @ current_R
        current_t = current_t + current_R @ t

        # Triangular puntos
        proj_matrix1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_matrix2 = np.hstack((current_R, current_t))

        points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convertimos de coordenadas homogéneas a 3D

        reconstructed_points.append(points_3d.T)

        if progress_callback:
            progress_callback((2 * len(keypoints_list) + i + 1) / total_steps)

    # Crear nube de puntos final
    reconstructed_points = np.vstack(reconstructed_points)  # Unimos todos los puntos 3D

    # Crear una malla directamente desde la nube de puntos
    print("Generando malla 3D a partir de la nube de puntos...")
    try:
        mesh = trimesh.Trimesh(points=reconstructed_points)
        mesh = trimesh.convex.convex_hull(mesh)  # Reconstrucción usando Convex Hull
        mesh.export(output_path)
        print(f"Malla 3D guardada en: {output_path}")
    except Exception as e:
        print("Error al generar la malla:", str(e))
        return

    if progress_callback:
        progress_callback(1.0)