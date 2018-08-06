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

"""tests for retina module"""

import os
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from scipy import ndimage
from skimage import color, filters, io
import cv2
from retipy import retina_grayscale

_resources = 'src/resources/images/'
_image_file_name = '01_h.jpg'
_image_path = _resources + _image_file_name


class TestRetinaGrayscale(TestCase):
    """Test class for Retina class"""

    def setUp(self):
        self.image = retina_grayscale.Retina_grayscale(None, _image_path)

    def test_constructor_invalid_path(self):
        """Test the retina constructor when the given path is invalid"""
        self.assertRaises(Exception, retina_grayscale.Retina_grayscale, None, _resources)

    def test_constructor_existing_image(self):
        """Test the constructor with an existing image"""
        image = retina_grayscale.Retina_grayscale(None, _image_path)
        none_constructor_image = retina_grayscale.Retina_grayscale(image.np_image, _image_file_name)

        assert_array_equal(image.np_image, none_constructor_image.np_image, "created images should be the same")

    def test_restore_mask(self):
        self.image.np_image[self.image.mask] = 5
        self.image.restore_mask()
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(self.image.np_image, original_image.np_image)

    def test_equalize_histogram(self):
        self.image.equalize_histogram()
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        original_image.np_image = clahe.apply(original_image.np_image)
        original_image.restore_mask()
        assert_array_equal(self.image.np_image, original_image.np_image)

    def test_opening(self):
        self.image.opening(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(self.image.np_image,
                           ndimage.grey_opening(original_image.np_image, size=(3, 3)))

    def test_top_hat(self):
        self.image.top_hat(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(self.image.np_image,
                           cv2.morphologyEx(original_image.np_image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (
                           3, 3))))

    def test_mean_filter(self):
        self.image.median_filter(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(self.image.np_image,
                           cv2.blur(original_image.np_image, (3, 3)))

    def test_gaussian_filter(self):
        self.image.gaussian_filter(17, 1.8)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(self.image.np_image,
                           cv2.GaussianBlur(self.np_image, (17, 17), 1.82))

    def test_median_filter(self):
        self.image.median_filter(3)
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(self.image.np_image,
                           cv2.medianBlur(original_image.np_image.astype(np.uint8), 3))

    def test_vessel_enhancement(self):
        self.image.homogenize()
        self.image.vessel_enhancement()
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        original_image.homogenize()
        original_image.np_image = abs(original_image.IH - 255)
        original_image.top_hat(40)
        assert_array_equal(self.image.IH, original_image.IH)

    def test_compare_with(self):
        self.image.threshold_image()
        original_image = retina_grayscale.Retina_grayscale(None, _image_path)
        assert_array_equal(
            self.image.compare_with(original_image).np_image,
            self.image.np_image - original_image.np_image,
            "image does not match")