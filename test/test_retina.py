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

import cv2
from numpy.testing import assert_array_equal
from retipy import retina

class TestRetina(TestCase):
    """Test class for Retina class"""

    _resources = 'resources'
    _image_file_name = 'im0001.png'
    _image_path = _resources + "/images/" + _image_file_name

    def setUp(self):
        self.image = retina.Retina(self._image_path)

    def tearDown(self):
        if os.path.isfile("./out_" + self._image_file_name):
            os.unlink("./out_" + self._image_file_name)

    def test_constructor_invalid_path(self):
        """Test the retina constructor when the given path is invalid"""
        self.assertRaises(retina.RetinaException, retina.Retina, self._resources)

    def test_segmented(self):
        """Test default value for segmented property"""
        self.assertEqual(
            False, self.image.segmented, "segmented should be false by default")
        self.image.segmented = True
        self.assertEqual(
            True, self.image.segmented, "segmented should be true")

    def test_threshold_image(self):
        self.image.threshold_image()
        _, opencv_output = cv2.threshold(cv2.imread(self._image_path), 127, 255, cv2.THRESH_BINARY)
        assert_array_equal(self.image.image, opencv_output, "segmented image does not match")

    def test_save_image(self):
        self.image.save_image(".")
        self.assertTrue(os.path.isfile("./out_" + self._image_file_name))
