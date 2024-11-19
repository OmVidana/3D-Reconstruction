import numpy as np
import cv2
import trimesh
from typing import List, Tuple


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


def load_images(image_paths: List[str]) -> List[np.ndarray]:
    """
    Carga las imágenes desde los archivos especificados.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {path}")
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir a RGB
    return images


def detect_features(images: List[np.ndarray]):
    """
    Detecta características clave (SIFT) en las imágenes.
    """
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(desc)
    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    """
    Empareja características entre dos imágenes usando BFMatcher.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def compute_fundamental_matrix(pts1: np.ndarray, pts2: np.ndarray):
    """
    Calcula la matriz fundamental entre dos conjuntos de puntos 2D.
    """
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    return F


def get_camera_poses(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extrae las posibles poses de cámara a partir de la matriz esencial.
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


def triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangula puntos 3D usando la técnica DLT (Direct Linear Transform).
    """
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d


def filter_points(points_3d: np.ndarray, max_dist: float = 100.0) -> np.ndarray:
    """
    Filtra puntos 3D que están demasiado lejos o son outliers.
    """
    distances = np.linalg.norm(points_3d, axis=1)
    mask = distances < max_dist
    return points_3d[mask]


def save_point_cloud(points_3d: np.ndarray, output_path: str):
    """
    Guarda la nube de puntos 3D en un archivo OBJ.
    """
    mesh = trimesh.points.PointCloud(points_3d)
    mesh.export(output_path)


def reconstruct_3d(image_paths: List[str], output_path: str) -> None:
    """
    Función para realizar la reconstrucción 3D a partir de una lista de imágenes.
    """
    images = load_images(image_paths)
    keypoints, descriptors = detect_features(images)

    all_points_3d = []
    camera_matrices = [np.hstack((np.eye(3), np.zeros((3, 1))))]  # Primera cámara en el origen

    for i in range(len(images) - 1):
        kp1 = keypoints[i]
        kp2 = keypoints[i + 1]
        desc1 = descriptors[i]
        desc2 = descriptors[i + 1]

        matches = match_features(desc1, desc2)

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

        # Normalizar los puntos
        pts1, T1 = normalize_points(pts1)
        pts2, T2 = normalize_points(pts2)

        # Calcular la matriz fundamental
        F = compute_fundamental_matrix(pts1, pts2)

        # Calcular la matriz esencial
        E = np.dot(T2.T, np.dot(F, T1))

        # Estimar las posibles poses de la cámara
        poses = get_camera_poses(E)

        # Triangulación de puntos 3D
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Primera cámara
        for R, t in poses:
            P2 = np.hstack((R, t.reshape(-1, 1)))  # Segunda cámara
            points_3d = triangulate_points(P1, P2, pts1, pts2)
            points_3d = filter_points(points_3d)
            all_points_3d.append(points_3d)

    # Unir todos los puntos 3D y guardarlos
    all_points_3d = np.vstack(all_points_3d)
    save_point_cloud(all_points_3d, output_path)
