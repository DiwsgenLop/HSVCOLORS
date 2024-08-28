import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Capturamos la cámara
parser = argparse.ArgumentParser()
parser.add_argument("index_camera", help="No encontrada", type=int)
args = parser.parse_args()

# Creamos el objeto para leer la cámara
capture = cv.VideoCapture(args.index_camera)

if not capture.isOpened():
    print("Error opening the camera")
    exit()

# Obtenemos dimensiones del frame
frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

# Configuración de la ventana de matplotlib
fig, (ax_mask, ax_video) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Inicialmente mostramos un frame vacío
frame = np.zeros((int(frame_height * 0.4), int(frame_width * 0.4), 3), dtype=np.uint8)
mask = np.zeros_like(frame)

img_mask = ax_mask.imshow(mask)
img_video = ax_video.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
ax_mask.set_title("MASK")
ax_video.set_title("VIDEO NORMAL")

# Creación de sliders
ax_min_h = plt.axes([0.1, 0.1, 0.35, 0.03])
ax_min_s = plt.axes([0.1, 0.15, 0.35, 0.03])
ax_min_v = plt.axes([0.1, 0.2, 0.35, 0.03])
ax_max_h = plt.axes([0.55, 0.1, 0.35, 0.03])
ax_max_s = plt.axes([0.55, 0.15, 0.35, 0.03])
ax_max_v = plt.axes([0.55, 0.2, 0.35, 0.03])

slider_min_h = Slider(ax_min_h, 'H MIN', 0, 179, valinit=0)
slider_min_s = Slider(ax_min_s, 'S MIN', 0, 255, valinit=0)
slider_min_v = Slider(ax_min_v, 'V MIN', 0, 255, valinit=0)
slider_max_h = Slider(ax_max_h, 'H MAX', 0, 179, valinit=179)
slider_max_s = Slider(ax_max_s, 'S MAX', 0, 255, valinit=255)
slider_max_v = Slider(ax_max_v, 'V MAX', 0, 255, valinit=255)

# Función de actualización cuando se mueven los sliders
def update(val):
    min_h = int(slider_min_h.val)
    min_s = int(slider_min_s.val)
    min_v = int(slider_min_v.val)
    max_h = int(slider_max_h.val)
    max_s = int(slider_max_s.val)
    max_v = int(slider_max_v.val)
    #Llamamos a la funcion para calcular el promedio y la desviacion estandar
    lh,ls,lv,hh,hs,hv = calculate_mean_std(min_h, min_s, min_v, max_h, max_s, max_v)
    #Hasta aqui guardamos los valores de los sliders
    low_values = np.array([min_h, min_s, min_v], dtype=np.uint8)
    high_values = np.array([max_h, max_s, max_v], dtype=np.uint8)
    low_values = np.array([ls, ls, lv])
    high_values = np.array([hh, hs, hv])
    #Esto deberia ser para otra funcion llamarla
    ret, frame = capture.read()
    if ret:
        frame = cv.resize(frame, (0, 0), fx=0.4, fy=0.4)
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame, low_values, high_values)
        
        img_mask.set_data(mask)
        img_video.set_data(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

#Funcion para calcular el promedio y la desviacion estandar
def calculate_mean_std(min_h, min_s, min_v, max_h, max_s, max_v):
    mean_h = (slider_min_h.val + slider_max_h.val) / 2
    mean_s = (slider_min_s.val + slider_max_s.val) / 2
    mean_v = (slider_min_v.val + slider_max_v.val) / 2

    std_h = (slider_max_h.val - slider_min_h.val) / 2
    std_s = (slider_max_s.val - slider_min_s.val) / 2
    std_v = (slider_max_v.val - slider_min_v.val) / 2

    low_values_h = mean_h - std_h
    low_values_s = mean_s - std_s
    low_values_v = mean_v - std_v

    high_values_h = mean_h + std_h
    high_values_s = mean_s + std_s
    high_values_v = mean_v + std_v
    
    #Devolvemos los valores menores y mayores de cada canal
    return low_values_h, low_values_s, low_values_v, high_values_h, high_values_s, high_values_v

# Llamaremos a la función si alguno de los botones se mueve
slider_min_h.on_changed(update)
slider_min_s.on_changed(update)
slider_min_v.on_changed(update)
slider_max_h.on_changed(update)
slider_max_s.on_changed(update)
slider_max_v.on_changed(update)

# Botón de reset
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color='gold', hovercolor='skyblue')

def resetSlider(event):
    slider_min_h.reset()
    slider_min_s.reset()
    slider_min_v.reset()
    slider_max_h.reset()
    slider_max_s.reset()
    slider_max_v.reset()

button.on_clicked(resetSlider)

# Bucle principal
while True:
    update(None)  # Actualizamos los valores al iniciar
    plt.pause(0.01)  # Espera un poco para que Matplotlib pueda actualizarse
    
    if cv.waitKey(1) & 0xFF == ord('q'):  # Salir con la tecla 'q'
        break

# Liberamos la cámara y cerramos las ventanas
capture.release()
cv.destroyAllWindows()
