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
import numpy as np
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
    :param greyscale: set the internal image as greyscale or rgb
    """
    def __init__(self, image, image_path, greyscale=True):
        if image is None:
            self.np_image = io.imread(image_path)
            _, file = path.split(image_path)
            self._file_name = file
        else:
            self.np_image = image
            self._file_name = image_path

        self.segmented = False
        self.old_image = None
        if greyscale:
            self.np_image = color.rgb2gray(self.np_image)
            self.depth = 1
        else:
            self.depth = 3
        self.shape = self.np_image.shape

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

    def reshape_square(self):
        """
        This function will normalise the image size, making a square with it and rounding the pixels:
        If the given image is 571 560, the new size will be 572 572, with zeroes in all new pixels.
        """
        max_value = self.shape[0] if self.shape[0] > self.shape[1] else self.shape[1]
        max_value = max_value + (max_value % 2)
        self.np_image = np.pad(
            self.np_image,
            ((0, max_value - self.shape[0]), (0, max_value - self.shape[1])),
            'constant',
            constant_values=(0, 0))
        self.shape = self.np_image.shape

    def get_window_sizes(self):
        """
        Returns an array with the possible window size that this image can be divided by without leaving empty space.
        584x584 would return [292,146,73]
        This is only available for square images (you can use reshape_square() before calling this method)
        :return: a list of possible window sizes.
        """
        sizes = []
        if self.shape[0] == self.shape[1]:
            current_value = self.shape[0]
            while current_value % 2 == 0:
                current_value = current_value // 2
                sizes.append(current_value)
        return sizes

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

    @property
    def filename(self):
        """Returns the filename of the retina image."""
        return self._file_name

    def _output_filename(self):
        return "/out_" + self.filename

    def save_image(self, output_folder):
        """Saves the image in the given output folder, the name will be out_<original_image_name>"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_folder + self._output_filename(), self.np_image)

    def view(self):  # pragma: no cover
        """show a window with the internal image"""
        io.imshow(self.np_image)
        plt.show()

    def compare_with(self, retinal_image):
        """
        Returns the difference between the given image and the stored one.

        :param retinal_image: the image to compare with
        :return:  a new Retina object with the differences
        """
        return Retina(self.np_image - retinal_image.np_image, "diff" + self.filename)


class Window(Retina):
    """
    a ROI (Region of Interest) that extends the Retina class
    TODO: Add support for more than depth=1 images (only if needed)
    """
    def __init__(self, image: Retina, dimension, method="separated", min_pixels=10):
        super(Window, self).__init__(
            image.np_image,
            image.filename)
        self.windows = Window.create_windows(image, dimension, method, min_pixels)
        if self.windows == []:
            raise(ValueError("No windows were created for the given retinal image"))
        else:
            self.shape = self.windows.shape
            self.mode = self.mode_pytorch
        self._tags = []

    @property
    def mode_pytorch(self):
        return "PYT"

    @property
    def mode_tensorflow(self):
        return "TF"

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        self._tags = value
        if value.shape[0] != self.shape[0]:
            raise ValueError("Wrong set of tags, expected {} got {}".format(self.shape[0], value.shape[0]))

    def switch_mode(self, mode):
        """
        Changes the internal window ordering depending on the given mode.
        Tensorflow style is [batch, width, height, depth]
        Pytorch style is [batch, depth, width, height]
        :param mode: new mode to change, can be self.tensorflow or self.pytorch
        """
        if mode == self.mode_pytorch and self.mode == self.mode_tensorflow:
            twin = np.swapaxes(self.windows, 2, 3)
            self.windows = np.swapaxes(twin, 1, 2)
            self.shape = self.windows.shape
            self.mode = self.mode_pytorch
        elif mode == self.mode_tensorflow and self.mode == self.mode_pytorch:
            twin = np.swapaxes(self.windows, 1, 2)
            self.windows = np.swapaxes(twin, 2, 3)
            self.shape = self.windows.shape
            self.mode = self.mode_tensorflow

    def _window_filename(self, window_id):
        return "out_w" + str(window_id) + "_" + self.filename

    def save_window(self, window_id, output_folder):
        """
        Saves the specified window in the given output folder, the name will be out_<original_image_name>
        :param window_id: the window_id to save
        :param output_folder: destination folder
        """
        if window_id >= self.windows.shape[0]:
            raise ValueError("Window value '{}' is more than allowed ({})".format(window_id, self.windows.shape[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(output_folder + self._window_filename(window_id), self.windows[window_id, 0])

    @staticmethod
    def create_windows(image: Retina, dimension, method="separated", min_pixels=10):
        """
        Creates multiple square windows of the given dimension for the current retinal image.
        Empty windows (i.e. only background) will be ignored

        Separated method will create windows of the given dimension size, that does not share any
        pixel, combined will make windows advancing half of the dimension, sharing some pixels
        between adjacent windows.
        :param image: an instance of Retina, to be divided in windows
        :param dimension:  window size (square of [dimension, dimension] size)
        :param method: method of separation (separated or combined)
        :param min_pixels: ignore windows with less than min_pixels with value. Set to zero to add all windows
        :return: a numpy array with the structure [window, depth, height, width]
        """
        if image.shape[0] != image.shape[1] or image.shape[0] % dimension != 0:
            raise ValueError(
                "image shape is not the same or the dimension value does not divide the image completely: "
                + "sx:{} sy:{} dim:{}".format(image.shape[0], image.shape[1], dimension))

        #                      window_count
        windows = []
        window_id = 0
        if method == "separated":
            windows = np.empty([(image.shape[0] // dimension) ** 2, image.depth, dimension, dimension])
            for x in range(0, image.shape[0], dimension):
                for y in range(0, image.shape[1], dimension):
                    t_window = image.np_image[x:(x + dimension), y:(y + dimension)]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = t_window
                        window_id += 1
        elif method == "combined":
            new_dimension = dimension // 2
            windows = np.empty([(image.shape[0] // new_dimension) ** 2, image.depth, dimension, dimension])
            if image.shape[0] % new_dimension != 0:
                raise ValueError(
                    "Dimension value '{}' is not valid, choose a value that its half value can split the image evenly"
                    .format(dimension))
            for x in range(0, image.shape[0] - new_dimension, new_dimension):
                for y in range(0, image.shape[1] - new_dimension, new_dimension):
                    t_window = image.np_image[x:(x + dimension), y:(y + dimension)]
                    if t_window.sum() >= min_pixels:
                        windows[window_id, 0] = image.np_image[x:(x + dimension), y:(y + dimension)]
                        window_id += 1
        if window_id <= windows.shape[0]:
            if window_id == 0:
                windows = []
            else:
                windows.resize([window_id, windows.shape[1], windows.shape[2], windows.shape[3]])

        #  print('created ' + str(window_id) + " windows")
        return windows


def detect_vessel_border(image: Retina, ignored_pixels=1):
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
        x_more = min(window.shape[0] - 1, pixel[0] + 1)
        y_more = min(window.shape[1] - 1, pixel[1] + 1)

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

        # remove all repeating x values
        current_x = -1
        filtered_vessel = []
        for pixel in vessel:
            if pixel[0] == current_x:
                pass
            else:
                filtered_vessel.append(pixel)
                current_x = pixel[0]

        vessel_x = []
        vessel_y = []
        for pixel in filtered_vessel:
            vessel_x.append(pixel[0])
            vessel_y.append(pixel[1])
        return [vessel_x, vessel_y]

    vessels = []
    for it_x in range(ignored_pixels, image.shape[0] - ignored_pixels):
        for it_y in range(ignored_pixels, image.shape[1] - ignored_pixels):
            if image.np_image[it_x, it_y] > 0:
                vessel = vessel_extractor(image, it_x, it_y)
                vessels.append(vessel)
    return vessels
