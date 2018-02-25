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

from retipy import retina

_resources = 'src/resources'
_image_file_name = 'img1.png'
_image_path = _resources + "/images/" + _image_file_name


class TestRetina(TestCase):
    """Test class for Retina class"""

    def setUp(self):
        self.image = retina.Retina(None, _image_path)

    def tearDown(self):
        if os.path.isfile("./out_" + _image_file_name):
            os.unlink("./out_" + _image_file_name)

    def test_constructor_invalid_path(self):
        """Test the retina constructor when the given path is invalid"""
        self.assertRaises(Exception, retina.Retina, None, _resources)

    def test_constructor_existing_image(self):
        """Test the constructor with an existing image"""
        image = retina.Retina(None, _image_path)
        none_constructor_image = retina.Retina(image.np_image, _image_file_name)

        assert_array_equal(image.np_image, none_constructor_image.np_image, "created images should be the same")

    def test_segmented(self):
        """Test default value for segmented property"""
        self.assertEqual(
            False, self.image.segmented, "segmented should be false by default")
        self.image.segmented = True
        self.assertEqual(
            True, self.image.segmented, "segmented should be true")

    def test_threshold_image(self):
        self.image.threshold_image()
        original_image = color.rgb2gray(io.imread(_image_path))
        output = original_image > filters.threshold_mean(original_image)

        assert_array_equal(self.image.np_image, output, "segmented image does not match")

    def test_apply_thinning(self):
        retina_image = retina.Retina(np.zeros((64, 64), np.uint8), _image_file_name)
        retina_image.np_image[10:17, 10:13] = 1
        retina_image.apply_thinning()
        output = [0, 1, 1, 1, 1, 0]
        assert_array_equal(retina_image.np_image[10:16, 11], output, "expected a line")

    def test_save_image(self):
        self.image.save_image(".")
        self.assertTrue(os.path.isfile("./out_" + _image_file_name))

    def test_undo(self):
        self.image.detect_edges_canny()
        original_image = retina.Retina(None, _image_path)
        self.assertRaises(
            AssertionError,
            assert_array_equal,
            self.image.np_image, original_image.np_image, "images should be different")
        self.image.undo()
        assert_array_equal(self.image.np_image, original_image.np_image, "image should be the same")

    def test_erode(self):
        self.image.threshold_image()
        self.image.erode(1)
        original_image = retina.Retina(None, _image_path)
        original_image.threshold_image()
        assert_array_equal(
            self.image.np_image, ndimage.binary_erosion(original_image.np_image, iterations=1))

    def test_dilate(self):
        self.image.threshold_image()
        self.image.dilate(1)
        original_image = retina.Retina(None, _image_path)
        original_image.threshold_image()
        assert_array_equal(
            self.image.np_image, ndimage.binary_dilation(original_image.np_image, iterations=1))

    def test_compare_with(self):
        self.image.threshold_image()
        original_image = retina.Retina(None, _image_path)
        assert_array_equal(
            self.image.compare_with(original_image).np_image,
            self.image.np_image - original_image.np_image,
            "image does not match")

    def test_reshape_image(self):
        self.image.reshape_square()
        self.assertEqual(self.image.shape[0], self.image.shape[1], "dimension should be the same when reshaping")

    def test_get_window_sizes(self):
        windows = self.image.get_window_sizes()
        assert_array_equal(windows, [], "window array does not match")
        self.image.reshape_square()
        windows = self.image.get_window_sizes()
        assert_array_equal(windows, [292,146,73], "window array does not match")


class TestWindow(TestCase):

    _image_size = 64

    def setUp(self):
        self._retina_image = retina.Retina(
            np.zeros((self._image_size, self._image_size), np.uint8), _image_file_name)

    def test_create_windows(self):
        # test with an empty image
        self.assertFalse(retina.Window(self._retina_image, 8).windows, "windows should be empty")

        # test with a full data image
        self._retina_image.np_image[:, :] = 1
        windows = retina.Window(self._retina_image, 8)
        self.assertEqual(windows.windows.shape[0], self._image_size, "expected 64 windows")

        # test with an image half filled with data
        self._retina_image.np_image[:, 0:int(self._image_size/2)] = 0
        windows = retina.Window(self._retina_image, 8)
        self.assertEqual(windows.windows.shape[0], self._image_size/2, "expected 32 windows")

    def test_create_windows_error_dimension(self):
        self.assertRaises(ValueError, retina.Window, self._retina_image, 7)

    def test_create_windows_combined(self):
        windows = retina.Window(self._retina_image, 8, "combined", 0)

        # combined should create (width/(dimension/2) - 1) * (height/(dimension/2) -1)
        # here is (64/4 -1) * (64/4 -1) = 225
        self.assertEqual(windows.windows.shape[0], 225, "there should be 225 windows created")

        windows = retina.Window(self._retina_image, 8, "combined")
        self.assertFalse(windows.windows, "no window should be created")

    def test_create_windows_combined_error_dimension(self):
        new_image = retina.Retina(np.zeros((66, 66), np.uint8), _image_file_name)
        self.assertRaises(ValueError, retina.Window, new_image, 33, "combined", 0)

    def test_vessel_extractor(self):
        self._retina_image.np_image[10, 10:20] = 1
        self._retina_image.np_image[11, 20] = 1
        self._retina_image.np_image[9, 20] = 1
        self._retina_image.np_image[11, 21] = 1
        self._retina_image.np_image[9, 21] = 1
        vessels = retina.detect_vessel_border(self._retina_image)

        self.assertEqual(len(vessels), 1, "only one vessel should've been extracted")
        self.assertEqual(len(vessels[0][0]), 14, "vessel should have 14 pixels")

    def test_output_filename(self):
        self._retina_image.np_image[:, :] = 1
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        window.save_window(1, "./")
        self.assertTrue(os.path.isfile(window._window_filename(1)), "file not found")
        os.unlink(window._window_filename(1))

    def test_switch_mode(self):
        window = retina.Window(self._retina_image, 8, min_pixels=0)
        assert_array_equal(window.shape, [64, 1, 8, 8], "window shape is incorrect")

        window.switch_mode(window.mode_tensorflow)
        assert_array_equal(window.shape, [64, 8, 8, 1], "window shape is incorrect")

        window.switch_mode(window.mode_pytorch)
        assert_array_equal(window.shape, [64, 1, 8, 8], "window shape is incorrect")
