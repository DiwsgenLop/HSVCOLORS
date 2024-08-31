# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from traitlets import Int

#Definimos nuestra camara por defecto es decir la camara 0
index_camera = 0

# We create a VideoCapture object to read from the camera (pass 0):
capture = cv.VideoCapture(index_camera)
# Check if the camera is opened correctly
if capture.isOpened() is False:
    print("Error opening the camera")
    exit()

# Obtener el ancho y alto del frame
frame_width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv.CAP_PROP_FRAME_WIDTH)
#fps = capture.get(cv.CAP_PROP_FPS)


# Crear la figura para los sliders y los ejes para mostrar imágenes
fig, (ax_slider, ax_img) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9)


# Creación de sliders
#Minimos de lado izquierdo
ax_min_h = plt.axes([0.1, 0.1, 0.35, 0.03])
ax_min_s = plt.axes([0.1, 0.15, 0.35, 0.03])
ax_min_v = plt.axes([0.1, 0.2, 0.35, 0.03])
#Maximos de lado derecho
ax_max_h = plt.axes([0.55, 0.1, 0.35, 0.03])
ax_max_s = plt.axes([0.55, 0.15, 0.35, 0.03])
ax_max_v = plt.axes([0.55, 0.2, 0.35, 0.03])

# Crear sliders para H, S y V
slider_min_h = Slider(ax_min_h, 'H min', 0, 179, valinit=100)
slider_min_s = Slider(ax_min_s, 'S min ', 0, 255, valinit=100)
slider_min_v = Slider(ax_min_v, 'V min', 0, 255, valinit=100)
slider_max_h = Slider(ax_max_h, 'H Max', 0, 179, valinit=100)
slider_max_s = Slider(ax_max_s, 'S Max', 0, 255, valinit=100)
slider_max_v = Slider(ax_max_v, 'V Max', 0, 255, valinit=100)


# Función para actualizar la imagen basada en los sliders
def update(val):
    # Leer los valores de los sliders
    h_min = int(slider_min_h.val)
    s_min = int(slider_min_s.val)
    v_min = int(slider_min_v.val)
    h_max = int(slider_max_h.val)
    s_max = int(slider_max_s.val)
    v_max = int(slider_max_v.val)

    # Leer el frame de la cámara
    ret, frame = capture.read()
    if ret is True:
        # Convertir el frame al espacio de color HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Crear rangos de valores para la segmentación
        low_values = np.array([h_min, s_min, v_min])
        high_values = np.array([h_max, s_max, v_max])

        # Crear la máscara para la imagen completa en HSV
        mask = cv.inRange(hsv_frame, low_values, high_values)
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)  # Convertir a RGB

        # Mostrar la imagen resultante en Matplotlib
        ax_slider.clear()
        ax_slider.imshow(mask_rgb)
        ax_slider.set_title("MASK")
        # Entrada de Camara
        ax_img.clear()
        ax_img.imshow(frame_rgb)
        ax_img.set_title("VIDEO NORMAL")
        fig.canvas.draw_idle()
    else:
        return

# Actualización con los sliders
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

# Mostrar la ventana de Matplotlib
plt.show(block=False)

try:
    while capture.isOpened():
        # Actualizar la imagen
        update(None)
        # Esperar 10 ms
        plt.pause(0.01)
        # Salir si se presiona la tecla 'q'
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
        #Si la ventana de matplotlib se cierra entonces se tiene que salir del bucle while
        if not plt.fignum_exists(fig.number):
            break
finally:
    # Asegurarse de liberar la cámara y cerrar todas las ventanas
    capture.release()
    cv.destroyAllWindows()
    plt.close('all')
    # Verificar si la cámara está liberada
    if capture.isOpened():
        print("Error: La cámara no se cerró correctamente.")
    else:
        print("Cámara cerrada exitosamente.")