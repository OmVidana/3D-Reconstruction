import time

import cv2
import numpy as np
import trimesh

# from PIL import ExifTags, Image
#
# def calibrate_images(image_paths):
#     """
#     Calibrar imágenes basándose en los metadatos EXIF del lente.
#     """
#     calibrated_images = []
#     for path in image_paths:
#         img = Image.open(path)
#         exif = img._getexif()
#         metadata = {ExifTags.TAGS.get(k): v for k, v in exif.items()} if exif else {}

#         # Calibración basada en datos de la lente
#         focal_length = metadata.get("FocalLength", None)
#         f_number = metadata.get("FNumber", None)
#         if focal_length and f_number:
#             print(f"Calibrando: {path} con Focal Length={focal_length} y F Number={f_number}")
#             # Si usas una librería para ajustes más complejos, agrégala aquí.
#         else:
#             print(f"Sin datos de lente: {path}, no se calibrará.")
#         calibrated_images.append(np.array(img))
#     return calibrated_images


def reconstruct_3d(image_paths, output_path, progress_callback=None):
    """
    Reconstrucción 3D a partir de múltiples imágenes y creación directa de una malla.
    """
    # Cargar imágenes
    images = []
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"Error al cargar la imagen: {path}")
            continue
        print(f"Imagen {idx} cargada: Tamaño = {img.shape}")
        images.append(img)

    if len(images) < 2:
        print("Se necesitan al menos dos imágenes válidas para la reconstrucción.")
        return

    # Calcular pasos totales para el progreso
    total_steps = len(images) + (len(images) - 1) + (len(images) - 1)  # Detección + emparejamiento + reconstrucción

    # Detectar características con SIFT
    sift = cv2.SIFT_create(nfeatures=5000)  # Detectar hasta 5000 características por imagen
    keypoints_list, descriptors_list = [], []

    for idx, img in enumerate(images):
        try:
            gray = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
            print(f"Imagen {idx}: {len(keypoints)} características detectadas")
            if progress_callback:
                progress_callback((idx + 1) / total_steps)
        except Exception as e:
            print(f"Error al procesar la imagen {idx}: {e}")

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

        print(f"Emparejamientos entre imágenes {i} y {i+1}: {len(good_matches)} coincidencias")
        if len(good_matches) < 10:
            print(f"Advertencia: Insuficientes coincidencias entre imágenes {i} y {i+1}")
        if progress_callback:
            progress_callback((len(keypoints_list) + i + 1) / total_steps)

    # Reconstrucción 3D iterativa
    reconstructed_points = []
    current_R = np.eye(3)  # Matriz de rotación inicial
    current_t = np.zeros((3, 1))  # Traslación inicial

    for i, matches in enumerate(all_matches):
        if len(matches) < 8:
            print(f"Saltando triangulación para imágenes {i} y {i+1} por insuficientes coincidencias")
            continue

        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches])

        try:
            E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, threshold=1.0)
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

            print(f"Puntos triangulados entre imágenes {i} y {i+1}: {points_3d.shape[1]}")
            if progress_callback:
                progress_callback((len(keypoints_list) + len(all_matches) + i + 1) / total_steps)
        except Exception as e:
            print(f"Error al reconstruir 3D entre imágenes {i} y {i+1}: {e}")

    # Crear nube de puntos final
    if len(reconstructed_points) == 0:
        print("No se generaron puntos 3D válidos.")
        return

    reconstructed_points = np.vstack(reconstructed_points)  # Unimos todos los puntos 3D

    # Filtrar puntos inválidos
    reconstructed_points = reconstructed_points[~np.isnan(reconstructed_points).any(axis=1)]
    reconstructed_points = reconstructed_points[~np.isinf(reconstructed_points).any(axis=1)]
    print(f"Puntos válidos finales: {reconstructed_points.shape[0]}")

    # Exportar la nube de puntos para depuración
    np.savetxt("nube_puntos.xyz", reconstructed_points, delimiter=" ", header="X Y Z", comments="")

    # Crear una malla directamente desde la nube de puntos
    print("Generando malla 3D a partir de la nube de puntos...")
    try:
        mesh = trimesh.Trimesh(vertices=reconstructed_points, process=False)
        mesh = trimesh.convex.convex_hull(mesh)  # Reconstrucción usando Convex Hull
        mesh.export(output_path)
        print(f"Malla 3D guardada en: {output_path}")
    except Exception as e:
        print("Error al generar la malla:", str(e))
        return

    if progress_callback:
        progress_callback(1.0)


def reconstruct_3d_mock(image_paths, output_path):
    """
    Mock function to simulate the behavior of reconstruct_3d.
    This simulates a long-running process without using a callback.
    """
    total_steps = 100  # Simulate 100 steps for simplicity
    for step in range(total_steps + 1):
        time.sleep(0.05)  # Simulate computation time
        # No progress callback is used anymore

    # Simulate successful output creation
    with open(output_path, "w") as output_file:
        output_file.write("# Mock 3D Model File\n")
        output_file.write(f"# Generated from {len(image_paths)} images\n")
        output_file.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n")

    print(f"Mock reconstruction complete. Output saved to: {output_path}")
