retipy
======
[![Build Status](https://travis-ci.org/alevalv/retipy.svg?branch=master)](https://travis-ci.org/alevalv/retipy)
[![Coverage Status](https://coveralls.io/repos/github/alevalv/retipy/badge.svg?branch=master)](https://coveralls.io/github/alevalv/retipy?branch=master)

The goal of this project is to create a python library to perform different operations on fundus retinal images.
The goal is to have vessel segmentation, vessel identification and tortuosity measures.

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
The library is also available as a docker container at [alevalv/retipy](https://hub.docker.com/r/alevalv/retipy/):
```bash
docker pull alevalv/retipy:latest
```

License
-------
retipy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
