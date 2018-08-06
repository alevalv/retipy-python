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

"""Module with common mathematical operators that could be reused elsewhere"""

import numpy as np
from retipy import retina

def derivative1_forward_h2(target, y):
    """
    Implements the taylor approach to calculate derivatives, giving the point of interest.
    The function used is y=f(target), where len(y)-2 > target >= 0
    :param target: the position to be derived
    :param y: an array of points with the values
    :return: the derivative of the target with the given values
    """
    if len(y) - 3 < target or target < 0:
        raise (ValueError("need two more points to calculate the derivative"))
    return (-y[target + 2] + 4 * y[target + 1] - 3 * y[target]) / 2


def derivative1_centered_h1(target, y):
    """
    Implements the taylor centered approach to calculate the first derivative.

    :param target: the position to be derived, must be len(y)-1 > target > 0
    :param y: an array with the values
    :return: the centered derivative of target
    """
    if len(y) - 1 <= target <= 0:
        raise (ValueError("Invalid target, array size {}, given {}".format(len(y), target)))
    return (y[target + 1] - y[target - 1]) / 2


def derivative2_centered_h1(target, y):
    """
    Implements the taylor centered approach to calculate the second derivative.

    :param target: the position to be derived,  must be len(y)-1 > target > 0
    :param y: an array with the values
    :return: the centered second derivative of target
    """
    if len(y) - 1 <= target <= 0:
        raise (ValueError("Invalid target, array size {}, given {}".format(len(y), target)))
    return (y[target + 1] - 2 * y[target] + y[target - 1]) / 4


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


def curve_to_image(x, y, filename="curve_image"):
    # get the maximum and minimum x and y values
    mm_values = np.empty([2, 2], dtype=np.int)
    mm_values[0, :] = 99999999999999
    mm_values[1, :] = -99999999999999
    for i in range(0, len(x)):
        if x[i] < mm_values[0, 0]:
            mm_values[0, 0] = x[i]
        if x[i] > mm_values[1, 0]:
            mm_values[1, 0] = x[i]
        if y[i] < mm_values[0, 1]:
            mm_values[0, 1] = y[i]
        if y[i] > mm_values[1, 1]:
            mm_values[1, 1] = y[i]
    distance_x = mm_values[1, 0] - mm_values[0, 0]
    distance_y = mm_values[1, 1] - mm_values[0, 1]
    # calculate which square with side 2^n of size will contain the line
    image_dim = 2
    while image_dim < distance_x or image_dim < distance_y:
        image_dim *= 2
    image_dim *= 2
    # values to center the
    padding_x = (mm_values[1, 0] - mm_values[0, 0]) // 2
    padding_y = (mm_values[1, 1] - mm_values[0, 1]) // 2

    image_curve = np.full([image_dim, image_dim], False)

    for i in range(0, len(x)):
        x[i] = x[i] - mm_values[0, 0]
        y[i] = y[i] - mm_values[0, 1]

    for i in range(0, len(x)):
        image_curve[x[i], y[i]] = True

    return retina.Retina(image_curve, filename)

