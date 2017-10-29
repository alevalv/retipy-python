#!/usr/bin/env python
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

from retipy import configuration, retina

CONFIG = configuration.Configuration("resources/retipy.config")

def select(parameter_list):
    pass

for filename in glob.glob(os.path.join(CONFIG.image_directory, '*.png')):
    segmentedImage = retina.Retina(None, filename)
