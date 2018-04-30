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

"""module to process operations related to vessels"""

import glob
import numpy as np
import os
from skimage import io


class Roi(object):
    """
    Class that represents a Region of interest

    Internally it will have an array of points which describes its area, this class needs at least
    two points to create a triangle Roi
    """
    def __init__(self, area_points: list, name: str = "Roi", description: str = ""):
        if len(area_points) < 2:
            raise ValueError("roi points must contains at least two points to form a polygon")
        for point in area_points:
            if len(point) != 2:
                raise ValueError("all points should be size 2, got {}".format(point))
        area_points.sort(key=lambda i: i[0])
        self.points = area_points
        self.name = name
        self.description = description

    def to_dict(self):
        """
        Transforms the internal data into a dictionary that can be used to serialise as json.
        :return: a dict with the Roi data.
        """
        out = {"name": self.name, "description": self.description, "roi_x": [], "roi_y": []}
        for point in self.points:
            out["roi_x"].append(point[0])
            out["roi_y"].append(point[1])
        return out

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


def process_bifurcation_dataset(path_to_dataset: str, file_filter: str = '*.png'):
    def _get_shared_pixels(red_channel: np.ndarray, green_channel: np.ndarray, s_x, s_y):
        shared_pixels = []
        if s_x > 0:
            for p in range(s_y, s_y+5):
                if red_channel[s_x-1, p] > 0 and green_channel[s_x-1, p] == 0:
                    iterator = s_x-1
                    sp_count = 5
                    while iterator >= 0 \
                            and red_channel[iterator, p] > 0 \
                            and green_channel[iterator, p] == 0:
                        sp_count -= 1
                        iterator -= 1
                    while sp_count >= 0:
                        shared_pixels.append([s_x + sp_count, p])
                        sp_count -= 1
        if s_y+6 < red_channel.shape[1]:
            for p in range(s_x, s_x + 5):
                if red_channel[p, s_y + 6] > 0 and green_channel[p, s_y + 6] == 0:
                    iterator = s_y + 6
                    sp_count = 5
                    while iterator < red_channel.shape[1] \
                            and red_channel[p, iterator] > 0 \
                            and green_channel[p, iterator] == 0:
                        sp_count -= 1
                        iterator += 1
                    while sp_count >= 0:
                        shared_pixels.append([s_x, s_y+5-sp_count])
                        sp_count -= 1
        if s_x + 6 < red_channel.shape[0]:
            for p in range(s_y, s_y + 5):
                if red_channel[s_x + 6, p] > 0 and green_channel[s_x + 6, p] == 0:
                    iterator = s_x + 6
                    sp_count = 5
                    while iterator < red_channel.shape[0] \
                            and red_channel[iterator, p] > 0 \
                            and green_channel[iterator, p] == 0:
                        sp_count -= 1
                        iterator += 1
                    while sp_count >= 0:
                        shared_pixels.append([s_x + sp_count, p])
                        sp_count -= 1
        return shared_pixels

    evaluations = []
    for filename in sorted(glob.glob(os.path.join(path_to_dataset, file_filter))):
        evaluation = \
            {
                "uri": filename,
                "data": [],
            }
        image = io.imread(filename)
        red_image = image[:, :, 0]
        green_image = image[:, :, 1]
        for i in range(0, red_image.shape[0]):
            for j in range(0, red_image.shape[1]):
                if red_image[i, j] > 0 and green_image[i, j] == 0:
                    shared_pixels = _get_shared_pixels(red_image, green_image, i, j)
                    evaluation["data"].append(
                        Roi([[i, j], [i+5, j], [i+5, j+5], [i, j+5]], "Bifurcation"))
                    # delete the bifurcation
                    green_image[i:i+5, j:j+5] = 255
                    # add the shared pixels
                    for pixel in shared_pixels:
                        green_image[pixel[0], pixel[1]] = 0
        evaluations.append(evaluation)
    return evaluations
