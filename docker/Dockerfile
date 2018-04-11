FROM centos:7

LABEL maintainer="Alejandro Valdes <alejandrovaldes@live.com>"

ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:$PATH

RUN yum update -y

RUN yum install -y wget git g++ bzip2 zsh libglib2.0-0 libsm6 libxrender1 libgl1 libhdf5-dev tk-devel tkinter && \
    yum clean all

RUN wget --no-check-certificate --quiet http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p ${CONDA_DIR} && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV NB_USER retipy
ENV NB_UID 1000
ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

RUN useradd -m -s /bin/zsh -N -u ${NB_UID} ${NB_USER} && \
    mkdir -p ${CONDA_DIR} && \
    chown ${NB_USER} ${CONDA_DIR} -R && \
    mkdir -p /src && \
    chown ${NB_USER} /src -R  && \
    mkdir -p /opt/retipy && \
    chown ${NB_USER} /opt/retipy -R

RUN localedef -i en_US -f UTF-8 en_US.UTF-8

USER retipy

ADD matplotlibrc /home/retipy/.config/matplotlib/matplotlibrc

RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install setuptools numpy matplotlib pillow scikit-image scipy h5py scikit-learn flask

ENV RETIPY_HOME /home/retipy/src

RUN git clone -b feature/json git://github.com/alevalv/retipy.git /opt/retipy && \
    pip install -e /opt/retipy/src/ && \
    mkdir ${RETIPY_HOME} && \
    cp /opt/retipy/src/*.py ${RETIPY_HOME} && \
    cp -r /opt/retipy/src/resources ${RETIPY_HOME} && \
    sed -i 's/\.\.\///' ${RETIPY_HOME}/resources/retipy.config && \
    mkdir ${RETIPY_HOME}/build && \
    rm ${RETIPY_HOME}/setup.py

ADD matplotlibrc ${RETIPY_HOME}/matplotlibrc

EXPOSE 5000

ENV PATH /opt/retipy/src/bin:$PATH

CMD ["/bin/sh", "/opt/retipy/src/bin/start_server.sh"]
