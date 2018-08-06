import numpy as np
from retipy import retina_grayscale
from PIL import Image
import cv2

base_directory = "../../../HRF/"
directory = base_directory+"images/01_h.jpg"

image = Image.open(directory)
image = retina_grayscale.Retina_grayscale(np.array(image), None)
print("Ecualizando histograma")
image.equalize_histogram()
print("Borrando reflejo central")
image.opening(12)
print("Corrigiendo sombras")
image.shadow_correction()
print("Homogenizando imagen")
image.homogenize()
print("Aplicando threshold")
IH = cv2.GaussianBlur(image.IH, (3, 3), 1.72).astype(np.uint8)
ret, normal_vessels_segmentation = cv2.threshold(IH,  0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
normal_vessels_segmentation = abs(255-normal_vessels_segmentation)
tiny_vessels_segmentation = cv2.adaptiveThreshold(IH, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
tiny_vessels_segmentation = abs(255-tiny_vessels_segmentation)
print(tiny_vessels_segmentation.dtype)

print("Borrando ruido")

_, npaContours, hierarchy = cv2.findContours(normal_vessels_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for npaContour in npaContours:
    if (cv2.contourArea(npaContour) < 150):
        cv2.drawContours(normal_vessels_segmentation, [npaContour], -1, (0, 0, 0), -1)

_, npaContours, hierarchy = cv2.findContours(tiny_vessels_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for npaContour in npaContours:
    if (cv2.contourArea(npaContour) < 150):
        cv2.drawContours(tiny_vessels_segmentation, [npaContour], -1, (0, 0, 0), -1)

final_vessels_segmentation = tiny_vessels_segmentation.astype(np.float) + normal_vessels_segmentation.astype(np.float)
k = final_vessels_segmentation >= 1
final_vessels_segmentation[k] = 255
k2 = final_vessels_segmentation <= 0
final_vessels_segmentation[k2] = 0

Image.fromarray((final_vessels_segmentation)).convert("RGB").save("../../../prueba/segmentation.png")
