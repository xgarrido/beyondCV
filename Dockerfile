FROM ubuntu:18.04
MAINTAINER Xavier Garrido <xavier.garrido@lal.in2p3.fr>

RUN apt-get update && apt-get install -y \
        build-essential                  \
        gfortran                         \
        git                              \
        python3                          \
        python3-pip                      \
        wget

RUN pip3 install git+https://github.com/xgarrido/cobaya.git
RUN pip3 install git+https://github.com/xgarrido/beyondCV.git

RUN cd /tmp/ && wget --no-check-certificate https://raw.githubusercontent.com/xgarrido/beyondCV/master/beyondCV.yaml
