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

"""tests for landamark module"""
from unittest import TestCase
from retipy.retina import Retina
from retipy import landmarks as l
import numpy as np
from numpy.testing import assert_array_equal


class TestLandmarks(TestCase):
    _resources = 'src/resources/images/'
    _image_file_name = 'img02.png'
    _image_path = _resources + _image_file_name

    def setUp(self):
        self.image = Retina(None, self._image_path)

    def test_potential_landmarks(self):
        self.image.threshold_image()
        self.image.skeletonization()
        skeleton = self.image.get_uint_image()
        result = l.potential_landmarks(skeleton, 3)
        landmarks = np.genfromtxt("src/resources/test_data/landmarks.csv", delimiter=',')

        assert_array_equal(result, landmarks, "landmark points does not match")



