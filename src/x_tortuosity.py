#!/usr/bin/env python3

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

"""
script to estimate the linear tortuosity of a set of retinal images, it will output the values
to a file in the output folder defined in the configuration. The output will only have the
estimated value and it is sorted by image file name.
"""

import argparse
import glob
import numpy as np
import os
import scipy.stats as stats
import sys

from retipy import configuration, retina, tortuosity

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c",
    "--configuration",
    help="the configuration file location",
    default="resources/retipy.config")
parser.add_argument(
    "-a",
    "--algorithm",
    help="sets the algorithm to run the tortuosity measure, options are: linear, fractal",
    default="linear"
)
_LINEAR_ALGORITHM = "linear"
_FRACTAL_ALGORITHM = "fractal"
args = parser.parse_args()

CONFIG = configuration.Configuration(args.configuration)

FILE_NAME = "{:03d}-{:02d}-{:02d}-{:.2f}.txt".format(
    CONFIG.window_size, CONFIG.pixels_per_window, CONFIG.sampling_size, CONFIG.r_2_threshold)

FILE = open(CONFIG.output_folder + '/' + FILE_NAME, 'w')

for filename in sorted(glob.glob(os.path.join(CONFIG.image_directory, '*.png'))):
    segmentedImage = retina.Retina(None, filename)
    segmentedImage.threshold_image()
    windows = retina.create_windows(segmentedImage, CONFIG.window_size, min_pixels=CONFIG.pixels_per_window)
    if args.algorithm == _LINEAR_ALGORITHM:
        vessel_count = 0
        positive_tortuous_vessels = 0
        for window in windows:
            window.apply_thinning()
            vessels = retina.detect_vessel_border(window)
            if vessels:
                for vessel in vessels:
                    # only check vessels of more than 6 pixels
                    if len(vessel[0]) > 6:
                        vessel_count += 1
                        if (tortuosity.linear_regression_tortuosity(vessel[0], vessel[1], CONFIG.sampling_size) <
                                CONFIG.r_2_threshold):
                            positive_tortuous_vessels += 1
        FILE.write("{:.2f}\n".format((positive_tortuous_vessels/vessel_count)*100))
    elif args.algorithm == _FRACTAL_ALGORITHM:
        fractal_dimensions = []
        for window in windows:
            #  window.apply_thinning()  //do we want to check on a skeleton?
            fractal_dimensions.append(tortuosity.fractal_tortuosity(window))
        average = np.average(fractal_dimensions)
        median = np.median(fractal_dimensions)
        mode = stats.mode(fractal_dimensions)
        FILE.write("{:.3f},{:.3f},{:.3f}\n".format(average, median, mode[0][0]))
    else:
        print("nothing to see here citizen, move along", file66=sys.stderr)

FILE.close()
