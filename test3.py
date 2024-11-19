import numpy as np
import cv2
import trimesh
from typing import List, Tuple
import os


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


def estimate_camera_poses(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima las poses de las cámaras a partir de puntos 2D y la matriz de calibración de la cámara.
    """
    # Matriz fundamental (se puede calcular con RANSAC o SVD)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

    # Descomponer la matriz esencial en rotación y traslación
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    return R, t


def triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangula puntos 3D usando la técnica DLT (Direct Linear Transform).
    """
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d


def save_xyz(points_3d: np.ndarray, output_path: str):
    """
    Guarda la nube de puntos 3D en un archivo .xyz.
    """
    with open(output_path, 'w') as f:
        for point in points_3d:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def save_obj(points_3d: np.ndarray, output_path: str):
    """
    Guarda la malla calculada en formato .obj.
    """
    # Suponemos que la nube de puntos es la malla o crear una malla de tipo 'PointCloud'
    mesh = trimesh.points.PointCloud(points_3d)
    mesh.export(output_path)


def reconstruct_3d(image_paths: List[str], output_path: str) -> None:
    """
    Reconstrucción 3D usando Structure from Motion (SfM) a partir de una lista de imágenes.
    """
    # Parámetros extraídos de los datos EXIF
    focal_length_mm = 4.0  # Longitud focal en mm
    sensor_width_mm = 6.5  # Ancho del sensor en mm (valor aproximado)
    image_width_px = 3060  # Resolución horizontal en píxeles
    image_height_px = 4080  # Resolución vertical en píxeles

    # Calcular la longitud focal en píxeles
    f_x = focal_length_mm * image_width_px / sensor_width_mm
    f_y = f_x  # Suponiendo que el píxel es cuadrado

    # Centro de la imagen
    c_x = image_width_px / 2
    c_y = image_height_px / 2

    # Crear la matriz de calibración K
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])

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

        # Estimar las poses de las cámaras
        R, t = estimate_camera_poses(pts1, pts2, K)

        # Formar la matriz de proyección de la primera cámara
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Primera cámara
        # Formar la matriz de proyección de la segunda cámara
        P2 = np.hstack((R, t.reshape(-1, 1)))  # Segunda cámara

        # Triangulación de puntos 3D
        points_3d = triangulate_points(P1, P2, pts1, pts2)

        all_points_3d.append(points_3d)

    # Unir todos los puntos 3D y guardar los archivos
    all_points_3d = np.vstack(all_points_3d)

    # Guardar la nube de puntos 3D en formato .xyz
    xyz_path = os.path.join(output_path, "point_cloud.xyz")
    save_xyz(all_points_3d, xyz_path)

    # Guardar la malla 3D calculada en formato .obj
    obj_path = os.path.join(output_path, "mesh.obj")
    save_obj(all_points_3d, obj_path)
