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

import csv
import numpy as np
from unittest import TestCase
from retipy import vessels
from retipy import tortuosity_measures as tm
from retipy.retina import Retina
from retipy.util import curve_to_image
from typing import List


class TestVessels(TestCase):
    _path_to_files = "src/resources/images/"
    _input_path = "src/test/input/"


    def test_process_bifurcation_dataset(self):
        bifurcations = vessels.process_bifurcation_dataset(self._path_to_files)
        self.assertEqual(20, len(bifurcations), "length of bifucartion does not match")
        for bifurcation in bifurcations:
            self.assertEqual(bifurcation["data"], [], "data is not empty")

    def test_get_vessel_by_bifurcation(self):
        _ii_b = "/home/alevalv/Code/maestria/Tesis/Datasets/DRIVE-Bifurcations/03_manual1_gt.png"
        _ii = "/home/alevalv/Code/maestria/Tesis/SourceCode/retipy/python/src/resources/images/img01.png"
        rois = vessels.process_bifurcation_image(_ii_b)
        image = vessels.RetinaRoi(_ii, rois=rois)
        image.threshold_image()
        image.skeletonize_1()
        image.view()
        image2 = vessels.RetinaRoi(_ii, rois=rois)
        image2.threshold_image()
        image2.apply_thinning()
        image2.view()

    def test_benchmark_rettort(self):
        a, v = self._load_rettort()
        results = np.empty([2, len(a), 5])
        for i in range(0, len(a)):
            results[0, i] = \
                [
                    tm.linear_regression_tortuosity(a[i]["x"], a[i]["y"]),
                    tm.distance_measure_tortuosity(a[i]["x"], a[i]["y"]),
                    tm.distance_inflection_count_tortuosity(a[i]["x"], a[i]["y"]),
                    tm.squared_curvature_tortuosity(a[i]["x"], a[i]["y"]),
                    tm.tortuosity_density(a[i]["x"], a[i]["y"])
                ]
            results[1, i] = \
                [
                    tm.linear_regression_tortuosity(v[i]["x"], v[i]["y"]),
                    tm.distance_measure_tortuosity(v[i]["x"], v[i]["y"]),
                    tm.distance_inflection_count_tortuosity(v[i]["x"], v[i]["y"]),
                    tm.squared_curvature_tortuosity(v[i]["x"], v[i]["y"]),
                    tm.tortuosity_density(v[i]["x"], v[i]["y"])
                ]

    def _load_rettort(self):
        _arteries_file = self._input_path + "arteries.csv"
        _veins_file = self._input_path + "veins.csv"
        arteries_img: List[Retina] = []
        veins_img: List[Retina] = []
        arteries: List[dict] = []
        veins: List[dict] = []
        with open(_arteries_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                x = []
                for point in row[1].split(' '):
                    x.append(int(point))
                y = []
                for point in row[2].split(' '):
                    y.append(int(point))
                gt = int(row[3])
                img = curve_to_image(x, y, row[0])
                arteries_img.append(img)
                arteries.append(
                    {
                        "name": row[0],
                        "x": x,
                        "y": y,
                        "gt": gt
                    })
        with open(_veins_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                x = []
                for point in row[1].split(' '):
                    x.append(int(point))
                y = []
                for point in row[2].split(' '):
                    y.append(int(point))
                gt = int(row[3])
                veins_img.append(curve_to_image(x, y, row[0]))
                veins.append(
                    {
                        "name": row[0],
                        "x": x,
                        "y": y,
                        "gt": gt
                    })
        return arteries, veins


