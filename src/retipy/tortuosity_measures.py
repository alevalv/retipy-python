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

"""Module with operations related to extracting tortuosity measures."""

import math
import numpy as np
from lib import fractal_dimension, smoothing
from retipy import math as m
from retipy.retina import Retina, Window, detect_vessel_border
from scipy.interpolate import CubicSpline


def _distance_2p(x1, y1, x2, y2):
    """
    calculates the distance between two given points
    :param x1: starting x value
    :param y1: starting y value
    :param x2: ending x value
    :param y2: ending y value
    :return: the distance between [x1, y1] -> [x2, y2]
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


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
    """
    This method detects the inflection points of a given curve y=f(x) by applying a convolution to
    the y values and checking for changes in the sign of this convolution, each sign change is
    interpreted as an inflection point.
    It will ignore the first and last 2 pixels.
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the array position in x of the inflection points.
    """
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for iterator in range(2, len(x)):
        if np.sign(cf[iterator]) != np.sign(cf[iterator - 1]):
            inflection_points.append(iterator - 1)
    return inflection_points


def _curve_to_image(x, y):
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

    return Retina(image_curve, "curve_image")


def linear_regression_tortuosity(x, y, sampling_size=6, retry=True):
    """
    This method calculates a tortuosity measure by estimating a line that start and ends with the
    first and last points of the given curve, then samples a number of pixels from the given line
    and calculates its determination coefficient, if this value is closer to 1, then the given
    curve is similar to a line.

    This method assumes that the given parameter is a sorted list.

    Returns the determination coefficient for the given curve
    :param x: the x component of the curve
    :param y: the y component of the curve
    :param sampling_size: how many pixels
    :param retry: if regression fails due to a zero division, try again by inverting x and y
    :return: the coefficient of determination of the curve.
    """
    if len(x) < 4:
        raise ValueError("Given curve must have more than 4 elements")
    try:
        min_point_x = x[0]
        min_point_y = y[0]

        slope = (y[len(y) - 1] - min_point_y)/(x[len(x) - 1] - min_point_x)

        y_intercept = min_point_y - slope*min_point_x

        sample_distance = max(round(len(x) / sampling_size), 1)

        # linear regression function
        def f_y(x1):
            return x1 * slope + y_intercept

        # calculate y_average
        y_average = 0
        item_count = 0
        for i in range(1, len(x) - 1, sample_distance):
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
            r_2 = linear_regression_tortuosity(y, x, retry=False)
        else:
            r_2 = 1  # mark not applicable vessels as not tortuous?
    if math.isnan(r_2):
        r_2 = 0
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
        raise ValueError("Given curve must have at least 2 elements")

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


def fractal_tortuosity(retinal_image: Retina):
    """
    Calculates the fractal dimension of the given image.
    The method used is the Minkowski-Bouligand dimension defined in
    https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension
    :param retinal_image:  a retinal image.
    :return: the fractal dimension of the given image
    """
    return fractal_dimension.fractal_dimension(retinal_image.np_image)


def fractal_tortuosity_curve(x, y):
    image = _curve_to_image(x, y)
    return fractal_dimension.fractal_dimension(image.np_image)


def tortuosity_density(x, y):
    """
    Defined in "A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity" by Grisan et al.
    DOI: 10.1109/IEMBS.2003.1279902

    :param x: the x points of the curve
    :param y: the y points of the curve
    :return: tortuosity density measure
    """
    inflection_points = _detect_inflection_points(x, y)
    n = len(inflection_points)
    if not n:
        return 0
    starting_position = 0
    sum_segments = 0
    # we process the curve dividing it on its inflection points
    for in_point in inflection_points:
        segment_x = x[starting_position:in_point]
        segment_y = y[starting_position:in_point]
        chord = _chord_length(segment_x, segment_y)
        if chord:
            sum_segments += _curve_length(segment_x, segment_y) / _chord_length(segment_x, segment_y) - 1
        starting_position = in_point

    return (n - 1)/n + (1/_curve_length(x, y))*sum_segments


def squared_curvature_tortuosity(x, y):
    """
    See Measurement and classification of retinal vascular tortuosity by Hart et al.
    DOI: 10.1016/S1386-5056(98)00163-4
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the squared curvature tortuosity of the given curve
    """
    curvatures = []
    x_values = range(1, len(x)-1)
    for i in x_values:
        x_1 = m.derivative1_centered_h1(i, x)
        x_2 = m.derivative2_centered_h1(i, x)
        y_1 = m.derivative1_centered_h1(i, y)
        y_2 = m.derivative2_centered_h1(i, y)
        curvatures.append((x_1*y_2 - x_2*y_1)/(y_1**2 + x_1**2)**1.5)
    return abs(np.trapz(curvatures, x_values))


def smooth_tortuosity_cubic(x, y):
    """
    TODO
    :param x: the list of x points of the curve
    :param y: the list of y points of the curve

    :return:
    """
    spline = CubicSpline(x, y)
    return spline(x[0])


def evaluate_window(window: Window, min_pixels_per_vessel=6, sampling_size=6, r2_threshold=0.80):  # pragma: no cover
    """
    Evaluates a Window object and sets the tortuosity values in the tag parameter.
    :param window: The window object to be evaluated
    :param min_pixels_per_vessel:
    :param sampling_size:
    :param r2_threshold:
    """
    tags = np.empty([window.shape[0], 7])
    # preemptively switch to pytorch.
    window.mode = window.mode_pytorch
    tft = fractal_tortuosity(window)
    for i in range(0, window.shape[0], 1):
        bw_window = window.windows[i, 0, :, :]
        retina = Retina(bw_window, "window{}" + window.filename)
        retina.threshold_image()
        retina.apply_thinning()
        vessels = detect_vessel_border(retina)
        vessel_count = 0
        t1, t2, t3, t4, td, tfi = 0, 0, 0, 0, 0, 0
        for vessel in vessels:
            if len(vessel[0]) > min_pixels_per_vessel:
                vessel_count += 1
                if linear_regression_tortuosity(vessel[0], vessel[1], sampling_size) > r2_threshold:
                    t1 += 1
                t2 += distance_measure_tortuosity(vessel[0], vessel[1])
                t3 += distance_inflection_count_tortuosity(vessel[0], vessel[1])
                t4 += squared_curvature_tortuosity(vessel[0], vessel[1])
                td += tortuosity_density(vessel[0], vessel[1])
                tfi += fractal_tortuosity_curve(vessel[0], vessel[1])
        if vessel_count > 0:
            t1 = t1/vessel_count
            t2 = t2/vessel_count
            t3 = t3/vessel_count
            td = td/vessel_count
        tags[i] = t1, t2, t3, t4, td, tfi, tft
    window.tags = tags
