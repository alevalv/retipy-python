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

"""tests for vessels module"""

from unittest import TestCase
from retipy import vessels


class TestVessels(TestCase):
    _path_to_files = "src/resources/images/"

    def test_process_bifurcation_dataset(self):
        bifurcations = vessels.process_bifurcation_dataset(self._path_to_files)
        self.assertEqual(20, len(bifurcations), "length of bifucartion does not match")
        for bifurcation in bifurcations:
            self.assertEqual(bifurcation["data"], [], "data is not empty")
