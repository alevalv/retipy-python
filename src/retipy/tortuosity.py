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


class TortuosityException(Exception):
    """Basic exception to showcase errors of the tortuosity module"""
    def __init__(self, message):
        super(TortuosityException, self).__init__(message)
        self.message = message


SAMPLING_SIZE = 6
R2_THRESHOLD = 0.98


def linear_regression_tortuosity(curve):
    """
    This method calculates a tortuosity measure by estimating a line that start and ends with the
    first and last points of the given curve, then samples a number of pixels from the given line
    and calculates its determination coefficient, if this value is closer to 1, then the given
    curve is similar to a line.

    This method assumes that the given parameter is a sorted list.

    Returns true if the curve has a determination coefficient that's lower than the threshold
    """
    if len(curve) < 4:
        raise TortuosityException("Given curve must have more than 4 elements")

    try:
        min_point = curve[0]
        max_point = curve[len(curve) - 1]

        slope = (max_point[1] - min_point[1])/(max_point[0] - min_point[0])

        y_intercept = min_point[1] - slope*min_point[0]

        sample_distance = round(len(curve) / SAMPLING_SIZE)

        # linear regression function
        def f_y(x):
            return x * slope + y_intercept

        # calculate y_average
        y_average = 0
        item_count = 0
        for i in range(1, len(curve) - 1, sample_distance):
            y_average += curve[i][1]
            item_count += 1
        y_average /= item_count

        # calculate determination coefficient
        top_sum = 0
        bottom_sum = 0
        for i in range(1, len(curve) - 1, sample_distance):
            top_sum += (f_y(curve[i][0]) - y_average)**2
            bottom_sum += (curve[i][1] - y_average)**2

        r_2 = top_sum / bottom_sum
    except ZeroDivisionError:  # mark not applicable vessels as not tortuous?
        r_2 = 1
    return r_2 < R2_THRESHOLD
