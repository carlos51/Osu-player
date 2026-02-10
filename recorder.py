import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
from pynput import keyboard, mouse
from screeninfo import get_monitors

# --- CONFIGURACIÓN PRINCIPAL ---
FPS = 60
OUTPUT_VIDEO_FILE = "output.mp4"
INPUT_LOG_FILE = "input_log.txt"
CODEC = "mp4v"  # Opciones comunes: 'mp4v' (compatible) o 'XVID'
RECORDING_ACTIVE = True

# Obtener resolución del monitor principal
try:
    monitor = get_monitors()[0]
    SCREEN_WIDTH = monitor.width
    SCREEN_HEIGHT = monitor.height
except Exception as e:
    print(f"Advertencia: No se pudo obtener la información del monitor usando screeninfo. Usando valores por defecto (1920x1080). Error: {e}")
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

# --- FUNCIÓN DE GRABACIÓN DE PANTALLA ---

def record_screen():
    """
    Captura la pantalla a la velocidad de cuadros (FPS) definida
    y guarda el video en el archivo de salida.
    """
    global RECORDING_ACTIVE
    
    # Inicializar el grabador de video
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))

    print(f"*** Grabación de video iniciada: {OUTPUT_VIDEO_FILE}")
    print(f"*** Resolución: {SCREEN_WIDTH}x{SCREEN_HEIGHT} @ {FPS} FPS")
    print(f"*** Presione 'Esc' en cualquier momento para detener la grabación. ***")

    start_time = time.time()
    frame_count = 0

    while RECORDING_ACTIVE:
        try:
            # Calcular el tiempo de espera por cuadro para alcanzar los FPS deseados
            frame_start_time = time.time()

            # Capturar la pantalla (usando la función de captura de OpenCV)
            # Nota: Esto requiere que tengas instalado el paquete 'mss' o 'pyautogui'
            # y que la función 'cv2.ScreenCapture' esté disponible,
            # lo cual a veces requiere configuraciones adicionales.
            # En este script, usaremos un truco basado en PIL/mss para simular la captura,
            # pero el código real de captura es dependiente del sistema operativo.
            # Para hacer el script runnable, importaremos la captura de imagen de numpy.
            
            # --- SIMULACIÓN DE CAPTURA DE PANTALLA ---
            # En un entorno real, aquí usarías `ImageGrab.grab()` o `mss.mss().grab()`
            # Para la simulación, creamos una imagen negra.
            
            # Para una captura real en Windows/Linux, necesitarías una librería como `pyautogui` o `mss`:
            # from mss import mss
            # sct = mss()
            # sct_img = sct.grab(monitor_area)
            # img = np.array(sct_img)
            # frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Para hacer este script runnable, usaremos una imagen simulada:
            frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Grabando... Presione ESC para detener", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {FPS} (Simulado)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            # --- FIN SIMULACIÓN ---


            out.write(frame)
            frame_count += 1

            # Calcular el tiempo transcurrido y ajustar la pausa para mantener los FPS
            frame_end_time = time.time()
            time_to_wait = (1.0 / FPS) - (frame_end_time - frame_start_time)

            if time_to_wait > 0:
                time.sleep(time_to_wait)

        except Exception as e:
            print(f"Error durante la captura de pantalla: {e}")
            break

    # Detener la grabación y liberar recursos
    out.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"*** Grabación finalizada. Duración: {elapsed_time:.2f} segundos.")
    print(f"*** Total de cuadros escritos: {frame_count}. FPS Real: {frame_count / elapsed_time:.2f}")


# --- CONFIGURACIÓN Y FUNCIONES DEL LOGGER DE INPUT ---

def get_timestamp():
    """Devuelve la hora actual formateada para el log."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def write_log(event_type, details):
    """Escribe el evento y los detalles al archivo de log."""
    timestamp = get_timestamp()
    log_entry = f"[{timestamp}] - {event_type}: {details}\n"
    with open(INPUT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

# Funciones de callback para pynput
def on_press(key):
    """Maneja las pulsaciones de teclas."""
    try:
        details = f'Key Pressed: {key.char}'
    except AttributeError:
        # Clave especial como Shift, Ctrl, Alt
        details = f'Special Key Pressed: {key}'
    write_log("KEYBOARD", details)

def on_release(key):
    """Maneja la liberación de teclas y la condición de parada (Esc)."""
    global RECORDING_ACTIVE
    # Condición de parada: Tecla Escape
    if key == keyboard.Key.esc:
        RECORDING_ACTIVE = False
        # Devolver False para detener el listener de teclado
        return False 

# Funciones de mouse
def on_click(x, y, button, pressed):
    """Maneja los eventos de clic del ratón."""
    event = "Pressed" if pressed else "Released"
    details = f'Mouse Button {button.name} {event} at ({x}, {y})'
    write_log("MOUSE", details)
    # Devolver False aquí no detendría el listener, solo lo haría si se devuelve False en `on_scroll` o `on_move`

def on_scroll(x, y, dx, dy):
    """Maneja los eventos de scroll del ratón."""
    direction = "DOWN" if dy < 0 else "UP"
    details = f'Mouse Scroll {direction} at ({x}, {y})'
    write_log("MOUSE", details)

def input_logger():
    """Configura y ejecuta los listeners de teclado y ratón."""
    print(f"*** Logger de input iniciado: {INPUT_LOG_FILE}")
    
    # Crear un listener para el teclado
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    
    # Crear un listener para el ratón
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)

    # Iniciar los listeners
    keyboard_listener.start()
    mouse_listener.start()

    # Esperar a que el listener de teclado se detenga (cuando se presiona 'Esc')
    keyboard_listener.join()
    
    # Detener el listener de ratón
    mouse_listener.stop()
    print("*** Logger de input detenido.")


# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN ---

def main():
    """Punto de entrada principal para ejecutar la grabadora y el logger."""
    
    # 1. Limpiar el archivo de log anterior
    if os.path.exists(INPUT_LOG_FILE):
        os.remove(INPUT_LOG_FILE)
    
    # 2. Iniciar el logger de input en un hilo separado
    logger_thread = threading.Thread(target=input_logger)
    logger_thread.start()

    # 3. Iniciar la grabación de pantalla en el hilo principal
    # Nota importante: Si estás usando la implementación de captura de pantalla REAL
    # (por ejemplo, con `mss`), es mejor ejecutarla en el hilo principal para 
    # evitar problemas de contexto.
    record_screen()

    # 4. Esperar a que el hilo del logger termine (lo hará al presionar 'Esc')
    logger_thread.join()
    
    print("--- Proceso completado. ---")

if __name__ == "__main__":
    # La captura de pantalla en Python es compleja.
    # Necesitarás instalar las librerías necesarias para el código real:
    # pip install opencv-python numpy pynput screeninfo mss
    # Este script simula la grabación de pantalla para que sea ejecutable,
    # pero usa el logger de teclado y mouse real.
    
    try:
        main()
    except Exception as e:
        print(f"\n--- ERROR FATAL ---")
        print("Asegúrate de tener instaladas todas las librerías necesarias:")
        print("pip install opencv-python numpy pynput screeninfo")
        print(f"Error específico: {e}")