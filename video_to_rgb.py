
import cv2
import torch
import numpy as np
import os

def video_a_tensores_por_batch_rgb(ruta_video, altura_deseada, ancho_deseado, tamano_batch, carpeta_salida):
    """
    Lee un video, redimensiona y guarda tensores de 3 CANALES (RGB/BGR) en disco por lotes (batches).
    
    :param ruta_video: Ruta al archivo de video.
    :param altura_deseada: Altura (H) en píxeles.
    :param ancho_deseado: Ancho (W) en píxeles.
    :param tamano_batch: Número de fotogramas por archivo guardado.
    :param carpeta_salida: Carpeta donde se guardarán los archivos .pt.
    :return: Lista con las rutas de los archivos generados.
    """
    
    # 1. Crear carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"📂 Carpeta creada: {carpeta_salida}")

    # 2. Abrir el video
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir {ruta_video}")
        return []

    lista_temporal = []
    nuevo_tamano = (ancho_deseado, altura_deseada)
    contador_batch = 0
    archivos_generados = []

    print(f"🔄 Procesando video con 3 canales... (Batch size: {tamano_batch})")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break # Fin del video

        # --- Procesamiento del fotograma ---
        # 1. Redimensionar (Mantiene 3 canales: H, W, 3)
        resized_frame = cv2.resize(frame, nuevo_tamano, interpolation=cv2.INTER_AREA)
        
        # *** CAMBIOS CLAVE AQUÍ ***

        # 2. Normalizar y convertir a Tensor PyTorch (de numpy a torch y [0, 255] a [0, 1])
        # La forma original de numpy/cv2 es (H, W, C)
        tensor_frame = torch.from_numpy(resized_frame).float() / 255.0

        # 3. Permutar ejes: Cambiar de (H, W, C) a (C, H, W) para PyTorch
        # Esto es vital para el formato estándar de tensores de imagen/video.
        # 
        tensor_frame = tensor_frame.permute(2, 0, 1) 
        # La forma final del fotograma individual es (3, H, W)
        
        # ----------------------------------

        # Agregar a la lista temporal
        lista_temporal.append(tensor_frame)

        # --- GUARDAR BATCH SI ESTÁ LLENO ---
        if len(lista_temporal) == tamano_batch:
            # 1. Apilar tensores (Batch_Size, C, H, W)
            tensor_batch = torch.stack(lista_temporal)
            
            # 2. Definir nombre del archivo
            nombre_archivo = os.path.join(carpeta_salida, f"batch_{contador_batch}.pt")
            
            # 3. Guardar en disco
            torch.save(tensor_batch, nombre_archivo)
            archivos_generados.append(nombre_archivo)
            
            print(f"   💾 Guardado: {nombre_archivo} | Forma: {tensor_batch.shape}")

            # 4. LIMPIAR MEMORIA y aumentar contador
            lista_temporal = [] 
            contador_batch += 1

    # --- GUARDAR FOTOGRAMAS RESTANTES (si sobran al final) ---
    if lista_temporal:
        tensor_batch = torch.stack(lista_temporal)
        nombre_archivo = os.path.join(carpeta_salida, f"batch_{contador_batch}.pt")
        torch.save(tensor_batch, nombre_archivo)
        archivos_generados.append(nombre_archivo)
        print(f"   💾 Guardado (Resto): {nombre_archivo} | Forma: {tensor_batch.shape}")

    cap.release()
    print("✅ Proceso finalizado.")
    return archivos_generados

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # 🛑 CONFIGURACIÓN
    # ¡IMPORTANTE! Reemplaza con la ruta de un video real
    RUTA_DEL_VIDEO = 'ruta/a/tu/video.mp4' 
    CARPETA_SALIDA = 'tensores_guardados_rgb'
    
    H, W = 128, 128    # Dimensiones (H: Altura, W: Ancho)
    BATCH_SIZE = 32    # Guardar cada 32 fotogramas en un archivo

    # Ejecutar función
    try:
        archivos = video_a_tensores_por_batch_rgb(RUTA_DEL_VIDEO, H, W, BATCH_SIZE, CARPETA_SALIDA)

        print(f"\nResumen: Se generaron {len(archivos)} archivos en '{CARPETA_SALIDA}'.")
        
        # Ejemplo de cómo cargar uno para verificar
        if archivos:
            primer_batch = torch.load(archivos[0])
            print(f"Verificación del primer archivo cargado: {primer_batch.shape}")
            print(f"El formato es (Batch, Canales, Altura, Ancho) -> ({primer_batch.shape[0]}, {primer_batch.shape[1]}, {primer_batch.shape[2]}, {primer_batch.shape[3]})")
    except RuntimeError as e:
        print(f"\n⚠️ Advertencia de prueba: Asegúrate de que '{RUTA_DEL_VIDEO}' exista para ejecutar el ejemplo.")
        print(f"Error específico: {e}")