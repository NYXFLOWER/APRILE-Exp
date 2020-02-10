# PoSePath

An optimization model for explaining the decision-making of the TIP-based multi-relational link prediction model.

### How To Use

- Download this repository, and change the working directory to its root directory
- Download example data and a trained model from web, and unzip it:
    - `wget https://www.dropbox.com/s/2vv3te17s1rfajw/data.zip `
    - `unzip data.zip`
- make sure the following packages have been installed with the latest version: `goatools`, `cython`, `pytorch`, `pytorch-geometric`, `matplotlib`, `scikit-learn`, `networkx`.
    - (Optitional) Create a test environment with Anaconda using `environment.yml` file.
- Run `python run.py -h` to see detailed instructions and examples. 
    - (Optitional) Run with GPU is recommanded (at least 20 times faster than using CPU only). 

An example of allocating a gpu node with interaction: 
- `salloc -p gpu --gres gpu:1 -c 10 --mem 40g -t 8:0:0`
