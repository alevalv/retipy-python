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

    def __init__(self, filePath):
        config = configparser.ConfigParser()
        config.read(filePath)
        if _PROPERTY_DEFAULT_CATEGORY not in config:
            raise ConfigurationException("General configuration not found in " + filePath)
        if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_IMAGE_DIRECTORY):
            self.image_directory = config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_IMAGE_DIRECTORY]
        if not self.image_directory:
            raise ConfigurationException("ImageDirectory is not configured")
        if config.has_option(_PROPERTY_DEFAULT_CATEGORY, _PROPERTY_WINDOW_SIZE):
            self.window_size = int(config[_PROPERTY_DEFAULT_CATEGORY][_PROPERTY_WINDOW_SIZE])
        if not self.window_size:
            raise ConfigurationException("WindowSize is not configured or is zero")
