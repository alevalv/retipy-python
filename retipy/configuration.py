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

"""Module for all configuration file related code"""

import configparser

_PROPERTY_DEFAULT_CATEGORY = 'General'
_PROPERTY_IMAGE_DIRECTORY = 'ImageDirectory'
_PROPERTY_WINDOW_SIZE = 'WindowSize'
_PROPERTY_PIXELS_PER_WINDOW = 'PixelsPerWindow'
_PROPERTY_SAMPLING_SIZE = "SamplingSize"
_PROPERTY_R2_THRESHOLD = "R2Threshold"


class ConfigurationException(Exception):
    """Basic exception to showcase errors of the configuration module"""
    def __init__(self, message):
        super(ConfigurationException, self).__init__(message)
        self.message = message


class Configuration(object):
    """
    Class that handles reading the configuration file and storing the values on it to access them
    later.
    """

    window_size = 0
    image_directory = 0
    pixels_per_window = 0
    sampling_size = 0
    r_2_threshold = 0

    def __init__(self, file_path):
        if file_path:
            config = configparser.ConfigParser()
            config.read(file_path)
            if _PROPERTY_DEFAULT_CATEGORY not in config:
                raise ConfigurationException(
                    _PROPERTY_DEFAULT_CATEGORY + "configuration not found in " + file_path)

            if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_IMAGE_DIRECTORY):
                self.image_directory = config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_IMAGE_DIRECTORY]
            if not self.image_directory:
                raise ConfigurationException(_PROPERTY_IMAGE_DIRECTORY + " is not configured")

            if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_WINDOW_SIZE):
                self.window_size = int(config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_WINDOW_SIZE])
            if not self.window_size:
                raise ConfigurationException(_PROPERTY_WINDOW_SIZE + "is not configured or is zero")

            if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_PIXELS_PER_WINDOW):
                self.pixels_per_window = int(
                    config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_PIXELS_PER_WINDOW])
            if not self.pixels_per_window:
                raise ConfigurationException(
                    _PROPERTY_PIXELS_PER_WINDOW + "is not configured or is zero")

            if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_SAMPLING_SIZE):
                self.sampling_size = int(
                    config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_SAMPLING_SIZE])
            if not self.sampling_size:
                raise ConfigurationException(
                    _PROPERTY_SAMPLING_SIZE + "is not configured or is zero")

            if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_R2_THRESHOLD):
                self.r_2_threshold = float(
                    config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_R2_THRESHOLD])
            if not self.r_2_threshold:
                raise ConfigurationException(
                    _PROPERTY_R2_THRESHOLD + "is not configured or is zero")
