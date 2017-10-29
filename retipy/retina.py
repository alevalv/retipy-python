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

import cv2

class RetinaException(Exception):
    """Basic exception to showcase errors of the retina module"""
    def __init__(self, message):
        super(RetinaException, self).__init__(message)
        self.message = message

class Retina(object):
    """
    Retina class that interally contains a matrix with the image data for a retinal image, it
    constructor expects a path to the image
    """
    def __init__(self, image, image_path):
        if image is None:
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise RetinaException("The given path is incorrect: " + image_path)
            _, file = path.split(image_path)
            self._file_name = file
        else:
            self.image = image
            self._file_name = image_path
        self.segmented = False


    def create_windows(self, dimension):
        """Creates multiple square windows of the given dimension for the current retinal image"""
        pass

    def threshold_image(self):
        """Applies a thresholding algorithm to the contained image."""
        _, threshold = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        self.image = threshold

    def _output_filename(self):
        return "/out_" + self._file_name

    def save_image(self, output_folder):
        """Saves the image in the given output folder, the name will be out_<original_image_name>"""
        cv2.imwrite(output_folder + self._output_filename(), self.image)

    def view(self):
        "show a window with the internal image"
        cv2.imshow(self._file_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Window(Retina):
    """a ROI (Region of Interest) that extends the Retina class"""
    def __init__(self, image, window_id, dimension, start_x, start_y):
        super(Window, self).__init__(
            image.image[start_x:(start_x + dimension), start_y:(start_y + dimension)],
            image._file_name) #pylint: disable=W0212
        self.window_id = window_id
        self._x = start_x
        self._y = start_y

    def _output_filename(self):
        return "out_w" + self.window_id + "_" + self._file_name
