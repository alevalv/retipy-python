# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Module with all REST endpoints to calculate tortuosity measures given a grayscale image
All operations are implemented as POST
"""

import base64
import flask
import io
from retipy.server import app
from retipy.server import base_url
import numpy as np
from retipy import retina_grayscale
from PIL import Image
import cv2

segmentation_url = base_url + "segmentation/"

@app.route(segmentation_url + "double_segmentation", methods=["POST"])
def post_segmentation_double_segmentation():
    data = {"success": False}

    if flask.request.method == "POST":
        json = flask.request.get_json(silent=True)
        if json is not None:  # pragma: no cover
            image = base64.b64decode(json["image"])
            image = Image.open(io.BytesIO(image))
            data = segment(image)
    return flask.jsonify(data)

def segment(image):
    image = retina_grayscale.Retina_grayscale(np.array(image), None)
    #print("Ecualizando histograma")
    image.equalize_histogram()
    #print("Borrando reflejo central")
    image.opening(12)
    #print("Corrigiendo sombras")
    image.shadow_correction()
    #print("Homogenizando imagen")
    image.homogenize()
    #print("Aplicando threshold")
    IH = cv2.GaussianBlur(image.IH, (3, 3), 1.72).astype(np.uint8)
    ret, normal_vessels_segmentation = cv2.threshold(IH, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    normal_vessels_segmentation = abs(255 - normal_vessels_segmentation)
    tiny_vessels_segmentation = cv2.adaptiveThreshold(IH, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    tiny_vessels_segmentation = abs(255 - tiny_vessels_segmentation)
    #print(tiny_vessels_segmentation.dtype)

    #print("Borrando ruido")

    _, npaContours, hierarchy = cv2.findContours(normal_vessels_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for npaContour in npaContours:
        if (cv2.contourArea(npaContour) < 150):
            cv2.drawContours(normal_vessels_segmentation, [npaContour], -1, (0, 0, 0), -1)

    _, npaContours, hierarchy = cv2.findContours(tiny_vessels_segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for npaContour in npaContours:
        if (cv2.contourArea(npaContour) < 150):
            cv2.drawContours(tiny_vessels_segmentation, [npaContour], -1, (0, 0, 0), -1)

    final_vessels_segmentation = tiny_vessels_segmentation.astype(np.float) + normal_vessels_segmentation.astype(
        np.float)
    k = final_vessels_segmentation >= 1
    final_vessels_segmentation[k] = 255
    k2 = final_vessels_segmentation <= 0
    final_vessels_segmentation[k2] = 0
    return image.get_base64_image(final_vessels_segmentation)