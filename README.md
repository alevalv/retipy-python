retipy-python
======
[![Build Status](https://travis-ci.org/alevalv/retipy-python.svg?branch=master)](https://travis-ci.org/alevalv/retipy-python)
[![Coverage Status](https://codecov.io/gh/alevalv/retipy-python/branch/master/graph/badge.svg)](https://codecov.io/gh/alevalv/retipy-python)

The goal of this project is to create a python library to perform different operations on fundus retinal images.
The goal is to have vessel segmentation, vessel identification and tortuosity measures available as a REST endpoints.

Currently a work in progress.

Installation
------------
### Development Environment
To use this project locally and be able to make changes to the retipy code, you can run the following command in
your console (having python3 and pip installed):

```bash
pip install --user -e .
```

This command should be ran inside the src folder that contains the retipy folder. It will make the retipy
library available to the user that ran it.
### Docker
The library is also available as a docker container at [alevalv/retipy-python](https://hub.docker.com/r/alevalv/retipy-python/):
```bash
docker pull alevalv/retipy-python:latest
```
By default, the docker image will expose a REST endpoint in port 5000.

License
-------
retipy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
