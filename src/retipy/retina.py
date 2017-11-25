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
import warnings

from scipy import ndimage
from skimage import color, feature, filters, io
from matplotlib import pyplot as plt
from lib import thinning


class Retina(object):
    """
    Retina class that internally contains a matrix with the image data for a retinal image, it
    constructor expects a path to the image

    :param image: a numpy array with the image data
    :param image_path: path to an image to be open
    """
    def __init__(self, image, image_path):
        if image is None:
            self.np_image = io.imread(image_path)
            _, file = path.split(image_path)
            self._file_name = file
        else:
            self.np_image = image
            self._file_name = image_path

        self.segmented = False
        self.size_x = self.np_image.shape[0]
        self.size_y = self.np_image.shape[1]
        self.old_image = None
        self.np_image = color.rgb2gray(self.np_image)
        self.depth = 1

##################################################################################################
# Image Processing functions

    def threshold_image(self):
        """Applies a thresholding algorithm to the contained image."""
        threshold = filters.threshold_mean(self.np_image)
        self.np_image = self.np_image > threshold
        self.depth = 1

    def detect_edges_canny(self, min_val=0, max_val=1):
        """
        Applies canny edge detection to the contained image. Fine tuning of the algorithm can be
        done using min_val and max_val
        """
        self._copy()
        self.np_image = feature.canny(self.np_image, low_threshold=min_val, high_threshold=max_val)

    def apply_thinning(self):
        """Applies a thinning algorithm on the stored image"""
        self._copy()
        self.np_image = thinning.thinning_zhang_suen(self.np_image)

    def erode(self, times):
        """
        Erodes the stored image
        :param times: number of times that the image will be eroded
        """
        self._copy()
        self.np_image = ndimage.binary_erosion(self.np_image, iterations=times)

    def dilate(self, times):
        """
        dilates the stored image
        :param times: number of times that the image will be dilated
        """
        self._copy()
        self.np_image = ndimage.binary_dilation(self.np_image, iterations=times)

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

    def filename(self):
        """Returns the filename of the retina image."""
        return self._file_name

    def _output_filename(self):
        return "/out_" + self._file_name

    def save_image(self, output_folder):
        """Saves the image in the given output folder, the name will be out_<original_image_name>"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_folder + self._output_filename(), self.np_image)

    def view(self):  # pragma: no cover
        """show a window with the internal image"""
        io.imshow(self.np_image)
        plt.show()


class Window(Retina):
    """a ROI (Region of Interest) that extends the Retina class"""
    def __init__(self, image, window_id, dimension, start_x, start_y):
        super(Window, self).__init__(
            image.np_image[start_x:(start_x + dimension), start_y:(start_y + dimension)],
            image.filename())
        self.window_id = window_id
        self._x = start_x
        self._y = start_y

    def _output_filename(self):
        return "out_w" + str(self.window_id) + "_" + self._file_name


def create_windows(image, dimension, method="separated", min_pixels=10):
    """
    Creates multiple square windows of the given dimension for the current retinal image.
    Empty windows (i.e. only background) will be ignored

    Separated method will create windows of the given dimension size, that does not share any
    pixel, combined will make windows advancing half of the dimension, sharing some pixels
    between adjacent windows.
    """
    windows = []
    window_id = 0

    if method == "separated":
        for x in range(0, image.size_x, dimension):
            for y in range(0, image.size_y, dimension):
                current_window = Window(image, window_id, dimension, x, y)
                pixel_values = current_window.np_image.sum()
                if pixel_values > min_pixels:
                    windows.append(current_window)
                    window_id += 1
    elif method == "combined":
        new_dimension = round(dimension/2)
        for x in range(0, image.size_x - new_dimension, new_dimension):
            for y in range(0, image.size_y - new_dimension, new_dimension):
                current_window = Window(image, window_id, dimension, x, y)
                pixel_values = current_window.np_image.sum()
                if pixel_values > min_pixels:
                    windows.append(current_window)
                    window_id += 1

    #  print('created ' + str(window_id) + " windows")
    return windows


def detect_vessel_border(image, ignored_pixels=1):
    """
    Extracts the vessel border of the given image, this method will try to extract all vessel
    borders that does not overlap.

    Returns a list of lists with the points of each vessel.

    :param image: the retinal image to extract its vessels
    :param ignored_pixels: how many pixels will be ignored from borders.
    """

    def neighbours(pixel, window):  # pragma: no cover
        """
        Creates a list of the neighbouring pixels for the given one. It will only
        add to the list if the pixel has value.

        :param pixel: the pixel position to extract its neighbours
        :param window:  the window with the pixels information
        :return: a list of pixels (list of tuples)
        """
        x_less = max(0, pixel[0] - 1)
        y_less = max(0, pixel[1] - 1)
        x_more = min(window.size_x - 1, pixel[0] + 1)
        y_more = min(window.size_y - 1, pixel[1] + 1)

        active_neighbours = []

        if window.np_image[x_less, y_less] > 0:
            active_neighbours.append([x_less, y_less])
        if window.np_image[x_less, pixel[1]] > 0:
            active_neighbours.append([x_less, pixel[1]])
        if window.np_image[x_less, y_more] > 0:
            active_neighbours.append([x_less, y_more])
        if window.np_image[pixel[0], y_less] > 0:
            active_neighbours.append([pixel[0], y_less])
        if window.np_image[pixel[0], y_more] > 0:
            active_neighbours.append([pixel[0], y_more])
        if window.np_image[x_more, y_less] > 0:
            active_neighbours.append([x_more, y_less])
        if window.np_image[x_more, pixel[1]] > 0:
            active_neighbours.append([x_more, pixel[1]])
        if window.np_image[x_more, y_more] > 0:
            active_neighbours.append([x_more, y_more])

        return active_neighbours

    def vessel_extractor(window, start_x, start_y):
        """
        Extracts a vessel using adjacent points, when each point is extracted is deleted from the
        original image
        """
        vessel = []
        pending_pixels = [[start_x, start_y]]
        while pending_pixels:
            pixel = pending_pixels.pop(0)
            if window.np_image[pixel[0], pixel[1]] > 0:
                vessel.append(pixel)
                window.np_image[pixel[0], pixel[1]] = 0

                # add the neighbours with value to pending list:
                pending_pixels.extend(neighbours(pixel, window))

        # sort by x position
        vessel.sort(key=lambda item: item[0])

        vessel_x = []
        vessel_y = []
        for pixel in vessel:
            vessel_x.append(pixel[0])
            vessel_y.append(pixel[1])
        return [vessel_x, vessel_y]

    vessels = []
    for it_x in range(ignored_pixels, image.size_x - ignored_pixels):
        for it_y in range(ignored_pixels, image.size_y - ignored_pixels):
            if image.np_image[it_x, it_y] > 0:
                vessel = vessel_extractor(image, it_x, it_y)
                vessels.append(vessel)
    return vessels
