# PoSePath
![Last-Commit](https://img.shields.io/github/last-commit/nyxflower/pose-path?style=plastic)
[![License](https://img.shields.io/github/license/nyxflower/pose-path?style=plastic)](https://github.com/NYXFLOWER/PoSe-Path/blob/master/LICENSE)
[![Releases](https://img.shields.io/github/v/release/nyxflower/pose-path?include_prereleases&style=plastic)](https://github.com/NYXFLOWER/PoSe-Path/releases)
![Size](https://img.shields.io/github/repo-size/nyxflower/pose-path?color=green&style=plastic)


PoSe-Path is based on the [GNNExplainer](http://papers.nips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks) (Ying et al., 2019). It is designed for explaining the predictions made by PoSe-Models. Here we use it to explore the relation between drug side effects and genes.

### How To Use

- Download this repository, and change the working directory to its root directory
- Download example data and a trained model from web, and unzip it:
    - `wget https://www.dropbox.com/s/khg3qrbh8klo4cf/data.zip `
    - `unzip data.zip`
- make sure the following packages have been installed with the latest version: `goatools`, `cython`, `pytorch`, `matplotlib`, `scikit-learn`, `networkx`, `pytorch-geometric`,  `pubchempy`.
    - (Optitional) Create a test environment with Anaconda using `environment.yml` file.
- Run `python run.py -h` to see detailed instructions and examples. 
    - (Optitional) Run with GPU is recommanded (at least 20 times faster than using CPU only). 

An example of allocating a gpu node with interaction: 
- `salloc -p gpu --gres gpu:1 -c 10 --mem 40g -t 8:0:0`

### Help for `run.py`

#### 1.  PoSePath Usage

`run.py drug_index_1 drug_index_2 side_effect_index regul_sore filter [-h] [-a] [-v]`

| Positional Arguments | Data Type        | Description                        |
| -------------------- | ---------------- | ---------------------------------- |
| drug_index_1         | int/int_list/str | [0, 283]                           |
| drug_index_2         | int/int_list/str | [0, 283]                           |
| side_effect_index    | int/int_list/str | [0, 860]                           |
| regul_sore           | float            | higher sore -> smaller pp-subgraph |
| filter               | float            | threshold probability              |

| Optional Arguments | Description                                    |
| ------------------ | ---------------------------------------------- |
| -h, --help         | show this help message and exit                |
| -a, --ifaddition   | add this flag for draw additional interactions |
| -v, --version      | show program's version number and exit         |

#### 2. Examples:

For the drug pair (D-88, D-95) and the side effect (SE-846) if prediction prob more than 0.7
```bash
$ python run.py 88 95 846 1 0.7
```
For all drug pairs causing the side effect (SE-846 and SE-848) (case: Prob > 0.99)
```bash
$ python run.py all all 846,848 2 0.99
```
For all side effects caused by the drug pairs (D-2, D-95), (D-2, D-107), (D-88, D-95), (D-88, D-107)
```bash
$ python run.py 2,88 95,107 2, all 2 0.89
```
For the side effects (SE-846 and SE-848) caused by all drug pairs which include the durg (D-88)
```bash
$ python run.py 88 all all 3 0.9
```

#### 3. Index Map:
​    See detailed drug, protein, side effect information in `../index-map` directory

#### 4. Output:
`exp_info.pkl` [dict]: 

- drug1, drug2, side_effect: list of int
-  pp_idx, pd_idx: int ndarray (2, n_edge), undirected
- pp_weight, pd_weight: float nparray (n_edges,), (0, 1]
- probability: list of float (0, 1)

`visual_explainer.png` [figure]: visualise pp_idx, pd_idx, pp_weight, pd_weight, drug1, drug2
`go_enrich.png, go_enrich_symbols.pdf` [figure]: visualise significant GO terms with/without gene symbols
`namespace.csv` [table]: associated significant GO terms' name, namespace, p-value


## License

This code is distributed under the terms of the [Apache License 2.0](https://github.com/blaisewang/img2latex-mathpix/blob/master/LICENSE).
