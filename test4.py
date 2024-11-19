import cv2
import numpy as np
import os
import trimesh

def detect_features(images):
    """
    Detecta puntos clave en las imágenes usando SIFT y devuelve sus descriptores.
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

def estimate_camera_poses(pts1, pts2):
    """
    Estima las poses relativas de las cámaras a partir de puntos 2D (sin matriz de cámara).
    Utiliza la descomposición de la matriz esencial para estimar la rotación y la traslación.
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
    return R, t

def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangula los puntos 3D utilizando DLT (Direct Linear Transform).
    """
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]  # Normaliza para obtener las coordenadas 3D
    return points_3d.T

def reconstruct_3d(image_filenames, output_path):
    """
    Flujo completo de SfM sin parámetros de cámara: detecta características, estima las posiciones relativas de
    las cámaras, triangula puntos 3D y guarda los resultados en .xyz y .obj.
    Recibe las imágenes como rutas absolutas.
    """
    # Cargar imágenes desde las rutas absolutas
    images = [cv2.imread(image_filename) for image_filename in image_filenames]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]  # Convertir a RGB

    # Detectar características (SIFT)
    keypoints, descriptors = detect_features(images)

    # Inicializar las matrices de cámaras (suponiendo que la primera cámara está en el origen)
    camera_matrices = [np.hstack((np.eye(3), np.zeros((3, 1))))]  # Primera cámara (en el origen)

    all_points_3d = []

    # Emparejar y procesar imágenes consecutivas
    for i in range(len(images) - 1):
        kp1 = keypoints[i]
        kp2 = keypoints[i + 1]
        desc1 = descriptors[i]
        desc2 = descriptors[i + 1]

        matches = match_features(desc1, desc2)

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

        # Estimar las poses relativas de las cámaras (rotación y traslación)
        R, t = estimate_camera_poses(pts1, pts2)

        # Primera cámara (identidad)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # Segunda cámara (con la rotación y traslación estimadas)
        P2 = np.hstack((R, t.reshape(-1, 1)))

        # Triangular los puntos 3D
        points_3d = triangulate_points(P1, P2, pts1, pts2)

        # Agregar los puntos 3D triangulados a la lista
        all_points_3d.append(points_3d)

    # Unir todos los puntos 3D
    all_points_3d = np.vstack(all_points_3d)

    # Crear la nube de puntos usando trimesh
    point_cloud = trimesh.points.PointCloud(all_points_3d)

    # Guardar la nube de puntos en formato .xyz
    xyz_path = os.path.join(output_path, 'point_cloud.xyz')
    np.savetxt(xyz_path, all_points_3d, fmt='%f %f %f')

    # Guardar la nube de puntos en formato .obj (sin malla, solo puntos)
    obj_path = os.path.join(output_path, 'point_cloud.obj')
    point_cloud.export(obj_path)

    print(f'Nube de puntos guardada en: {xyz_path} y {obj_path}')
