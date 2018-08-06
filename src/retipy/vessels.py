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
from matplotlib import pyplot as plt
from skimage import io
from typing import List

from retipy.retina import Retina


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


class RetinaRoi(Retina):
    """
    Class that extends Retina and adds a region of interest image, providing a convenient method
    is_in_roi(x, y) to verify if the given pixel is inside of one of the region of interest
    """
    def __init__(self, path: str, image: np.ndarray = None, rois: List[Roi] = ()):
        super().__init__(image, path)
        self.threshold_image()
        self.apply_thinning()
        self.rois = rois
        self.np_roi = np.full(self.shape, False)
        for roi in rois:
            x = roi.points[0]
            y = roi.points[3]
            self.np_roi[x[0]:y[0], x[1]:y[1]] = True

    def is_in_roi(self, x: int, y: int) -> bool:
        """
        checks if the given coordinate is inside a roi. If the coordinate is invalid, False is
        returned
        :param x: the x coordinate of the pixel
        :param y: the y coordinate of the pixel
        :return: True if the pixel is inside a Roi, False otherwise
        """
        if 0 <= x < self.np_roi.shape[0] and 0 <= y < self.np_roi.shape[1]:
            return self.np_roi[x, y]
        else:
            return False

    def view(self):  # pragma: no cover
        tmp_image = np.empty([self.np_image.shape[0], self.np_image.shape[1], 3])
        tmp_image[:, :, 0] = self.np_image
        tmp_image[:, :, 1] = self.np_roi
        tmp_image[:, :, 2] = 0
        io.imshow(tmp_image)
        plt.show()

    def view_roi(self):  # pragma: no cover
        io.imshow(self.np_roi)
        plt.show()


def get_vessels_by_bifurcation(retina: Retina, rois: List[Roi]):
    def _process_vessel(c_retina: RetinaRoi, x, y):
        to_process = [[x, y]]
        processed = []
        while len(to_process) > 0:
            current_pixel = to_process.pop(0)
            if c_retina.np_image[current_pixel[0], current_pixel[1]] > 0:
                processed.append(current_pixel)
                c_retina.np_image[current_pixel[0], current_pixel[1]] = 0

        return processed

    retina_roi: RetinaRoi = RetinaRoi(retina.filename, retina.np_image, rois)
    retina_roi.threshold_image()
    retina_roi.apply_thinning()
    for i in range(0, retina_roi.shape[0]):
        for j in range(0, retina_roi.shape[1]):
            if retina_roi.np_image[i, j]:
                _process_vessel(retina_roi, i, j)
                # TODO: finish this


def process_bifurcation_image(image_path: str) -> List[Roi]:
    """
    Process a retinal image that is already segmentated  and has marked Rois as bright red
    rectangles.
    :param image_path: the path to the image to extract is Roi
    :return: a List of Roi of the input image, empty if no Roi are found within the image
    """
    def _get_shared_pixels(red_channel: np.ndarray, green_channel: np.ndarray, s_x, s_y):
        """
        This function checks for cases when there are Rois that shares pixels, in those cases when
        we extract one of the Roi from the given image, we must restore the shared pixels. This
        function detects which pixels are shared and returns them in a simple array. This function
        assumes that the Roi are in a red square, so the green channel will have no values when is
        inside a Roi, but the red channel will have it.

        It is important to note that this function will assume that the starting point is the top
        left pixel of the Roi.
        :param red_channel: a numpy array with the red channel of the image
        :param green_channel: a numpy array with the green chanel of the image
        :param s_x: the starting x point of the Roi
        :param s_y: the starting y point of the Roi
        :return: an array with the shared pixels, if there are no shared pixels, an empty array
                 will be returned
        """
        out_pixels = []
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
                        out_pixels.append([s_x + sp_count, p])
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
                        out_pixels.append([s_x, s_y+5-sp_count])
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
                        out_pixels.append([s_x + sp_count, p])
                        sp_count -= 1
        return out_pixels

    rois: List[Roi] = []
    image = io.imread(image_path)
    red_image = image[:, :, 0]
    green_image = image[:, :, 1]
    for i in range(0, red_image.shape[0]):
        for j in range(0, red_image.shape[1]):
            if red_image[i, j] > 0 and green_image[i, j] == 0:
                shared_pixels = _get_shared_pixels(red_image, green_image, i, j)
                rois.append(
                    Roi([[i, j], [i + 5, j], [i + 5, j + 5], [i, j + 5]], "Bifurcation"))
                # delete the bifurcation
                green_image[i:i + 5, j:j + 5] = 255
                # add the shared pixels
                for pixel in shared_pixels:
                    green_image[pixel[0], pixel[1]] = 0
    return rois


def process_bifurcation_dataset(path_to_dataset: str, file_filter: str = '*.png'):
    """
    Process the DRIVE bifurcation dataset, extracting only the data related to the Rois.
    :param path_to_dataset: the path to the dataset
    :param file_filter: the image type, by default is '*.png'
    :return: a dictionary containing all data from the dataset, to faciliate its convertion to a
            json payload
    """
    evaluations = []
    for filename in sorted(glob.glob(os.path.join(path_to_dataset, file_filter))):
        evaluation = \
            {
                "uri": filename,
                "data": process_bifurcation_image(filename),
            }
        evaluations.append(evaluation)
    return evaluations
