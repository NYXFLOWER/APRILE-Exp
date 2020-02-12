# PoSePath
[![Releases](https://img.shields.io/github/v/release/nyxflower/pose-path?include_prereleases&style=flat-square)](https://github.com/NYXFLOWER/PoSe-Path/releases)
[![Downloads](https://img.shields.io/github/downloads/nyxflower/pose-path/total?color=green&style=flat-square)](https://github.com/NYXFLOWER/PoSe-Path/releases)
[![License](https://img.shields.io/github/license/nyxflower/pose-path?style=flat-square)](https://github.com/NYXFLOWER/PoSe-Path/blob/master/LICENSE)

An optimization model for explaining the decision-making of the TIP-based multi-relational link prediction model.

### How To Use

- Download this repository, and change the working directory to its root directory
- Download example data and a trained model from web, and unzip it:
    - `wget https://www.dropbox.com/s/2vv3te17s1rfajw/data.zip `
    - `unzip data.zip`
- make sure the following packages have been installed with the latest version: `goatools`, `cython`, `pytorch`, `matplotlib`, `scikit-learn`, `networkx`, `pytorch-geometric`,  `pubchempy`.
    - (Optitional) Create a test environment with Anaconda using `environment.yml` file.
- Run `python run.py -h` to see detailed instructions and examples. 
    - (Optitional) Run with GPU is recommanded (at least 20 times faster than using CPU only). 

An example of allocating a gpu node with interaction: 
- `salloc -p gpu --gres gpu:1 -c 10 --mem 40g -t 8:0:0`

## License

This code is distributed under the terms of the [Apache License 2.0](https://github.com/blaisewang/img2latex-mathpix/blob/master/LICENSE).