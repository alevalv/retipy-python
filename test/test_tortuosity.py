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

"""tests for tortuosity module"""

from unittest import TestCase

from retipy import tortuosity


class TestTortuosity(TestCase):
    _straight_line = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]

    def test_linear_regression_tortuosity(self):
        self.assertFalse(
            tortuosity.linear_regression_tortuosity(self._straight_line),
            "a straight line should return false")

    def test_linear_regression_tortuosity_error_size(self):
        self.assertRaises(
            tortuosity.TortuosityException,
            tortuosity.linear_regression_tortuosity,
            [[1, 1]])

    def test_linear_regression_tortuosity_no_interpolation(self):
        self.assertFalse(
            tortuosity.linear_regression_tortuosity([[1, 1], [2, 1], [3, 1], [4, 1]]),
            "a not applicable line should return false")