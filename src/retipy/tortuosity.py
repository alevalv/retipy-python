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

"""Module with operations related to extracting tortuosity measures."""

import cmath
import numpy as np


class TortuosityException(Exception):
    """Basic exception to showcase errors of the tortuosity module"""
    def __init__(self, message):
        super(TortuosityException, self).__init__(message)
        self.message = message


SAMPLING_SIZE = 6


def _distance_2p(x1, y1, x2, y2):
    """
    calculates the distance between two given points
    :param x1: starting x value
    :param y1: starting y value
    :param x2: ending x value
    :param y2: ending y value
    :return: the distance between [x1, y1] -> [x2, y2]
    """
    return cmath.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _curve_length(x, y):
    """
    calculates the length(distance) of the given curve, iterating from point to point.
    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the curve length
    """
    distance = 0
    for i in range(0, len(x) - 1):
        distance += _distance_2p(x[i], y[i], x[i + 1], y[i + 1])
    return distance


def _chord_length(x, y):
    """
    distance between starting and end point of the given curve

    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the chord length of the given curve
    """
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])


def _detect_inflection_points(x, y):
    df = np.diff(y)
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for iterator in range(1, len(x)):
        if np.sign(cf[iterator]) != np.sign(cf[iterator - 1]):
            inflection_points.append(x[iterator - 1])
    return inflection_points


def linear_regression_tortuosity(x, y, retry=True):
    """
    This method calculates a tortuosity measure by estimating a line that start and ends with the
    first and last points of the given curve, then samples a number of pixels from the given line
    and calculates its determination coefficient, if this value is closer to 1, then the given
    curve is similar to a line.

    This method assumes that the given parameter is a sorted list.

    Returns the determination coefficient for the given curve
    :param x: the x component of the curve
    :param y: the y component of the curve
    :param retry: if regression fails due to a zero division, try again by inverting x and y
    """
    if len(x) < 4:
        raise TortuosityException("Given curve must have more than 4 elements")
    try:
        min_point_x = x[0]
        min_point_y = y[0]

        slope = (y[len(y) - 1] - min_point_y)/(x[len(x) - 1] - min_point_x)

        y_intercept = min_point_y - slope*min_point_x

        sample_distance = round(len(x) / SAMPLING_SIZE)

        # linear regression function
        def f_y(x1):
            return x1 * slope + y_intercept

        # calculate y_average
        y_average = 0
        item_count = 0
        for i in range(0, len(x), sample_distance):
            y_average += y[i]
            item_count += 1
        y_average /= item_count

        # calculate determination coefficient
        top_sum = 0
        bottom_sum = 0
        for i in range(1, len(x) - 1, sample_distance):
            top_sum += (f_y(x[i]) - y_average) ** 2
            bottom_sum += (y[i] - y_average) ** 2

        r_2 = top_sum / bottom_sum
    except ZeroDivisionError:
        if retry:
            #  try inverting x and y
            r_2 = linear_regression_tortuosity(y, x, False)
        else:
            r_2 = 1  # mark not applicable vessels as not tortuous?
    return r_2


def distance_measure_tortuosity(x, y):
    """
    Distance measure tortuosity defined in:
    William E Hart, Michael Goldbaum, Brad Côté, Paul Kube, and Mark R Nelson. Measurement and
    classification of retinal vascular tortuosity. International journal of medical informatics,
    53(2):239–252, 1999.

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the arc-chord tortuosity measure
    """
    if len(x) < 2:
        raise TortuosityException("Given curve must have at least 2 elements")

    return _curve_length(x, y) / _chord_length(x, y)


def distance_inflection_count_tortuosity(x, y):
    """
    Calculates the tortuosity by using arc-chord ratio multiplied by the curve inflection count
    plus 1

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the inflection count tortuosity
    """
    return distance_measure_tortuosity(x, y) * (len(_detect_inflection_points(x, y)) + 1)
