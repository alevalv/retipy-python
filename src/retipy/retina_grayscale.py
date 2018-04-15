# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Alejandro Valdes
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

"""retina module to handle basic image processing on retinal images"""

from os import path
from copy import copy
import numpy as np
import warnings
import cv2
import math

from scipy import ndimage, signal
from skimage import io
from matplotlib import pyplot as plt
from PIL import Image


class Retina_grayscale(object):
    """
    Retina_grayscale class that internally contains a matrix with the green channel image data for a retinal image, it
    constructor expects a path to the image

    :param image_path: path to an fundus image to be open
    :param result_image_path: path to an  to be open
    :param mask_path: path to an fundus image to be open
    """
    @staticmethod
    def _open_image(img_path):
        return io.imread(img_path)

    def __init__(self, image_path, result_image_path=None, mask_path=None):
        self.np_image = io.imread(image_path)
        _, file = path.split(image_path)
        self._file_name = file
        self.np_image = self.np_image[:, :, 1]
        self.old_image = None
        self.shape = self.np_image.shape

        self.result_image = io.imread(result_image_path)
        self.gray255_to_bin(self.result_image)

        self.mask = io.imread(mask_path)
        self.fe = np.zeros((np.sum(self.mask == True), 15))

##################################################################################################
# Image Processing functions

    def restore_mask(self):
        """Restores the mask when it has been affected by the application of a filter"""
        if self.mask is not None:
            self.np_image[self.mask == 0] = 0

    def gray255_to_bin(self, target):
        """Converts a binary image with grayscale values (0 and 255) to binary values (0 and 1)"""
        mask = target == 255
        target[mask] = 1

    def equalize_histogram(self):
        """Applies contrast limited adaptive histogram equalization algorithm (CLAHE)"""
        self._copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.np_image = clahe.apply(self.np_image)
        self.restore_mask()

    def opening(self, size_structure):
        """
        dilates and erodes the stored image, by default the structure is a cross
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = ndimage.grey_opening(self.np_image, size=(size_structure, size_structure))

    def top_hat(self, size_structuring_element):
        """
        Applies Top-hat filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = cv2.morphologyEx(self.np_image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (size_structuring_element, size_structuring_element)))

    def mean_filter(self, structure):
        """
        Applies mean filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = signal.medfilt(self.np_image, structure)

    def gaussian_filter(self, structure, sigma):
        """
        Applies Gaussian filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = cv2.GaussianBlur(self.np_image, (structure, structure), sigma)

    def median_filter(self, structure):
        """
        Applies median filter
        :param size_structure: size of kernel to apply in the filter
        """
        self._copy()
        self.np_image = ndimage.median_filter(self.np_image, size=(structure, structure))

    def shadow_correction(self):
        """Applies the following filters: mean filter with a 3x3 kernel, Gaussian filter with a kernel of 9x9
        and sigma of 1.82, and a median filter with a 60x60 kernel. The resulting image is subtracted with the
        original image and finally the values obtained from the subtraction are moved to the 256 possible grayscale values"""
        self._copy()
        minuendo = np.copy(self.np_image)
        self.mean_filter(3)
        self.gaussian_filter(9, 1.82)
        mean_value = np.mean(self.np_image)
        self.np_image[self.mask] = mean_value
        self.median_filter(40)
        self.np_image = minuendo - self.np_image
        min = self.np_image.min()
        self.np_image = self.np_image - min
        max = self.np_image.max()
        escala = float(255)/(max)
        for row in range(0, self.shape[0]):
            for col in range(0, self.shape[1]):
                self.np_image[row, col] = int(self.np_image[row, col] * escala)
        self.restore_mask()

    def homogenize(self):
        """Moves all the values resulting from the correction of the shadows to the possible 255 values"""
        self._copy()
        g_input_max = self.np_image.max()
        aux = np.zeros(self.shape)
        for row in range(0, self.shape[0]):
            for col in range(0, self.shape[1]):
                g = self.np_image[row, col] + 180 - g_input_max
                if (g < 0):
                    aux[row, col] = 0
                elif (g > 255):
                    aux[row, col] = 255
                else:
                    aux[row, col] = g
        self.IH = np.copy(aux)

    def vessel_enhancement(self):
        """Generates a new vessel-enhanced image"""
        self.np_image = abs(self.IH - 255)
        self.top_hat(15)

    def extract_features(self):
        """Extracts the following features: The features since 1 to 4 are features based on gray levels and the features
            since 5 to 12 are based on moment invariants. Finaly, the features are saved and the class of the pixels,
            the position of the pixels are saved too
        """
        contador = 0
        for row in range(9, self.shape[0] - 8):
            for col in range(9, self.shape[1] - 8):
                if (self.mask[row, col] == 1):

                    region = self.IH[row - 4:row + 4, col - 4:col + 4]
                    pixel = self.IH[row, col]
                    f1 = pixel - np.min(region)
                    f2 = np.max(region) - pixel
                    f3 = pixel - np.mean(region)
                    f4 = np.std(region)

                    IV = self.np_image[row - 9:row + 8, col - 9:col + 8]
                    IHU = np.dot(IV, cv2.GaussianBlur(IV, (17, 17), 1.72))
                    moments = cv2.HuMoments(cv2.moments(IHU)).flatten()
                    f6 = abs(math.log(moments[0])) if moments[0] > 0 else 100.0
                    f7 = abs(math.log(moments[1])) if moments[1] > 0 else 100.0
                    f8 = abs(math.log(moments[2])) if moments[2] > 0 else 100.0
                    f9 = abs(math.log(moments[3])) if moments[3] > 0 else 100.0
                    f10 = abs(math.log(moments[4])) if moments[4] > 0 else 100.0
                    f11 = abs(math.log(moments[5])) if moments[5] > 0 else 100.0
                    f12 = abs(math.log(moments[6])) if moments[6] > 0 else 100.0

                    self.fe[contador, :] = [f1, f2, f3, f4, self.IH[row, col], f6, f7, f8, f9, f10, f11, f12,
                                            self.result_image[row, col], row, col]
                    contador = contador + 1

##################################################################################################
# I/O functions

    def _copy(self):
        self.old_image = copy(self.np_image)

    def undo(self):
        """
        Reverts the latest modification to the internal image, useful if you are testing different
         values
        """
        self.np_image = self.old_image

    @property
    def filename(self):
        """Returns the filename of the retina image."""
        return self._file_name

    def _output_filename(self):
        return "/out_" + self.filename

    def save_image(self, output_folder):
        """Saves the image in the given output folder, the name will be out_<original_image_name>"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_folder + self._output_filename(), self.np_image)

    def view(self):  # pragma: no cover
        """show a window with the internal image"""
        io.imshow(self.np_image)
        plt.show()

    def compare_with(self, retinal_image):
        """
        Returns the difference between the given image and the stored one.

        :param retinal_image: the image to compare with
        :return:  a new Retina object with the differences
        """
        return Retina_grayscale(self.np_image - retinal_image.np_image, "diff" + self.filename)

