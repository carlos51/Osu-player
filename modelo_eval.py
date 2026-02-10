import time
import threading
import cv2
import mss
import numpy as np
from pynput import mouse, keyboard
import torch
from collections import deque
from modelo import CNN_LSTM_Model as MiModelo

# --- CONFIGURACIÓN DE MODELO ---
#checkpoint  = "C:/Users/carlo/Proyects/Python/osu player/modelos/molelo15.pth" 
checkpoint  = "C:/Users/carlo/Proyects/Python/osu player/modelos/m2/molelo130.pth" 
USAR_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USAR_CUDA else "cpu")

# --- CONFIGURACIÓN DE PANTALLA ---
REGION_CONFIG = {
    "top": 46,
    "left": 304,
    "width": 1312,
    "height": 1024
}

# --- CONFIGURACIÓN DE ENTRADA ---
FPS_OBJETIVO = 30.0
INTERVALO_OBJETIVO = 1.0 / FPS_OBJETIVO
BUFFER_SIZE = 32  # Cantidad de frames necesarios
IMG_SIZE = 128    # Tamaño de entrada del modelo

# --- VARIABLES GLOBALES ---
program_running = True
inference_active = False


# --- CARGAR MODELO ---
print(f"Cargando modelo en {DEVICE}...")
try:

    #model = MiModelo(3,4,1,64,2*20) #modelo_1
    model = MiModelo(3,8,1,64,2*20) # modelo15

    checkpoint = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(DEVICE)
    model.eval() # Modo evaluación (apaga dropout, batchnorm, etc.)rww
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error cargando modelo (asegúrate de que la ruta sea correcta): {e}")
    program_running = False

# --- FUNCIONES DE MOUSE Y TECLADO ---
mouse_controller = mouse.Controller()

def on_press(key):
    global inference_active, program_running
    try:
        char = key.char
        if char == 'r': # 'r' para arrancar la IA
            inference_active = not inference_active
            estado = "ACTIVADA" if inference_active else "PAUSADA"
            print(f"\n>>> IA {estado} <<<")
        elif char == 'w': # 'w' para cerrar todo
            print("\n>>> CERRANDO PROGRAMA <<<")
            program_running = False
            return False
    except AttributeError:
        pass


def smooth_trajectory(points):
    points = np.array(points)
    n = len(points)
    
    # Pesos gaussianos centrados
    t = np.linspace(-1, 1, n)
    weights = np.exp(-4 * t**2)
    weights /= weights.sum()
    
    return np.sum(points * weights[:, None], axis=0)


# --- PROCESAMIENTO DE IMAGEN ---

def preprocesar_frame(img_bgra):
    """Convierte frame capturado de BGRA a RGB 128x128 con 3 canales."""
    
    # 1. Quitar canal Alpha y pasar a RGB (3 canales)
    # OpenCV por defecto usa BGR, así que convertimos de BGRA a RGB
    frame_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
    
    # 2. Redimensionar a 128x128
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    
    # 3. Normalizar (0 a 1)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    
    # 4. Cambiar el orden de dimensiones (H, W, C) -> (C, H, W)
    frame_final = np.transpose(frame_norm, (2, 0, 1))
    
    return frame_final

# --- BUCLE PRINCIPAL DE INFERENCIA ---
def run_inference_loop():
    global program_running
    
    sct = mss.mss()
    monitor = REGION_CONFIG
    
    # Buffer para guardar los últimos 32 frames
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    trajectory_buffer = deque(maxlen=BUFFER_SIZE)  # 32 pasados + 20 futuros

    
    print(f"Capturando región: {monitor}")
    print("Presiona 'r' para activar la IA.")

    with torch.no_grad(): # No necesitamos gradientes para inferencia
        while program_running:
            start_time = time.time()

            if inference_active:
                # 1. Captura de pantalla
                sct_img = sct.grab(monitor)
                img_np = np.array(sct_img)
                
                # 2. Preprocesar y guardar en buffer
                processed_frame = preprocesar_frame(img_np)
                frame_buffer.append(processed_frame)

                # 2.1 Guardar trayectoria
                x, y = mouse_controller.position
                trajectory_buffer.append([x/1920,y/1080])
                
                # 3. Solo ejecutamos si tenemos 32 frames
                if len(frame_buffer) == BUFFER_SIZE:
                    # Convertir buffer a numpy: (32, 128, 128)
                    input_np = np.array(frame_buffer)
                    points = np.array(trajectory_buffer)
                    
                    # Convertir a Tensor y pasar a GPU/CPU
                    input_tensor = torch.from_numpy(input_np).to(DEVICE)
                    
                    # Darle la forma exacta: (Batch=1, Time=32, Channels=1, H=128, W=128)
                    # input_tensor tiene forma (32, 128, 128)
                    # view(1, 32, 1, 128, 128)
                    input_tensor = input_tensor.view(1, BUFFER_SIZE, 3, IMG_SIZE, IMG_SIZE)
                    
                    # --- INFERENCIA ---
                    outputs = model(input_tensor)
                    
                    # Asumiendo que outputs es un tensor tipo [x, y, click]
                    # y que x, y están normalizados (0 a 1)
                    outputs = outputs.cpu().numpy()[0] # Sacar del batch y llevar a CPU
                    preds = outputs.reshape(-1, 2)  # shape (20, 2)

                    #print(points.shape,preds.shape)

                    points = np.concatenate([points,preds])

                    # trayectorias suavizadas
                    smooth_x, smooth_y = smooth_trajectory(points)
                    
                    T = 2
                    pred_x, pred_y, pred_click = outputs[0+T], outputs[1+T], 0

                    
                    
                    # --- EJECUTAR ACCIÓN ---
                    

                    target_x = int((pred_x * 1920))
                    target_y = int((pred_y * 1080))

                    #target_x = int(smooth_x * 1920)
                    #target_y = int(smooth_y * 1080)

                                        
                    #print(target_x,target_y)
                    #print(smooth_x,smooth_y)
                    # Mover Mouse
                    mouse_controller.position = (target_x, target_y)
                    
                    # Hacer Click (Asumiendo que click > 0.5 es un click)
                    if pred_click > 0.5: 
                        mouse_controller.press(mouse.Button.left)
                        mouse_controller.release(mouse.Button.left)
                        # print(f"CLICK en {target_x}, {target_y}", end='\r')
                    
                
                # Control de FPS (para no ir más rápido de lo que el juego renderiza)r
                process_time = time.time() - start_time
                wait_time = INTERVALO_OBJETIVO - process_time
                if wait_time > 0:
                    time.sleep(wait_time)
            else:
                time.sleep(0.1)

# --- INICIO ---
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    run_inference_loop()
except Exception as e:
    print(f"\nError fatal: {e}")
finally:
    listener.stop()
    print("Fin del programa.")