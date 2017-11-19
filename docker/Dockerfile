FROM ubuntu:16.04

LABEL maintainer="Alejandro Valdes <alejandrovaldes@live.com>"

ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git g++ bzip2 zsh libglib2.0-0 libsm6 libxrender1 libgl1 libhdf5-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate --quiet http://repo.continuum.io/miniconda/Miniconda3-4.3.27-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-4.3.27-Linux-x86_64.sh -f -b -p ${CONDA_DIR} && \
    rm Miniconda3-4.3.27-Linux-x86_64.sh

ENV NB_USER retipy
ENV NB_UID 1000

RUN useradd -m -s /bin/zsh -N -u ${NB_UID} ${NB_USER} && \
    mkdir -p ${CONDA_DIR} && \
    chown ${NB_USER} ${CONDA_DIR} -R && \
    mkdir -p /src && \
    chown ${NB_USER} /src -R

USER retipy

ADD matplotlibrc /home/keras/.config/matplotlib/matplotlibrc

RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install setuptools numpy matplotlib pillow scikit-image scipy

RUN git clone git://github.com/alevalv/retipy.git /src/retipy && \
    pip install -e /src/retipy/src && \
    cp /src/retipy/src/*.py /src && \
    rm -rf /src/retipy && \
    rm /src/setup.py

WORKDIR /src
