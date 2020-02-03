#!/usr/bin/env bash

# install mini-conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# set up channels for biopython
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge

# install packages
conda install goatools
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

