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

"""vessel extraction script"""

import os
import glob

from retipy import configuration, retina, tortuosity

CONFIG = configuration.Configuration("src/resources/retipy.config")

for filename in glob.glob(os.path.join(CONFIG.image_directory, '*.png')):
    segmentedImage = retina.Retina(None, filename)
    segmentedImage.threshold_image()
    windows = retina.create_windows(segmentedImage, CONFIG.window_size)
    vessel_count = 0
    positive_tortuous_vessels = 0
    for window in windows:
        window.apply_thinning()
        vessels = retina.detect_vessel_border(window)
        if vessels:
            for vessel in vessels:
                # only check vessels of more than 6 pixels
                if len(vessel) > 6:
                    vessel_count += 1
                    if tortuosity.linear_regression_tortuosity(vessel):
                        positive_tortuous_vessels += 1

    print("{:.2f}".format((positive_tortuous_vessels/vessel_count)*100))
