# PoSePath

## How To Use

1. Download the source files using the following command:
```bash
wget https://www.dropbox.com/s/7xk5p7db3d9ij7l/PoSePath.zip
```
2. After downloading, unzip the file and change the working directory to the `PoSePath` directory.

3. For an interactive job on `cac` using GPUs, run the following (this example uses 1 GPU, 10 CPUs, and 40GB of memory for 8 hours):
```bash
salloc -p gpu --gres gpu:1 -c 10 --mem 40g -t 8:0:0
```

4. interprete the drug-protein, protein-protein paths for polypharmacy side effect prediction. Run the following (this example for drug(indexed-88), drug(indexed-95) and side effect(indexed-846) with a regularization score of 2)
```bash
python run.py 88 95 846 2
```
  - A higher regularization score returns a smaller graph
  - use `python run.py -h` for detailed expalination and help
  - the results are saved in `fig` (for a visulized graphs - [[example]](https://github.com/Flower-Mt/PoSePath/blob/master/README.md)) and `pkl`(for a `dict` object) directories
  - index maps for drug and protein are in the `index-map` directory
  - results in `pool` directory help to select (drug, drug, side_effect) triples to analyse
  
  ## About file name

* `-[32, 64, 64]-128`: network structure
* `-record file`: side-effect-wise score of auprc, auroc, ap scores of the model on test set
* `-layerwise file`: side-effect-wise and layer-wise (explained below) scores of the model train set
* `-edgewise file`: list all layer-wise scores for all drug pairs and side effects
* `-with_p_feat`: protein features are parameters (tunable)
* `-with_d_p_feat`: protein and drug features are parameters (tunable)
* `all-side_effect`: averaged scores for different models (see explanation below)

## About column names

#### In `*-record.csv`:

* `side effect`: name of each side effect

* `side_effect_index`: index of each side effect in the code

* `n_instance`: the number of appearance of each side-effect in dataset

* `auprc` / `auroc` / `ap`: auprc, auprc, ap

#### In `*-layerwise.csv` files:

* `auprc-0`: the auprc score when the protein feature is the only input to the PD network

* `auprc-1`: the auprc score when the protein feature and the output of 1st GCN are the input to the PD network

* `auprc-2`: the auprc score when the protein feature, the output of 1st and 2nd GCNs are the input to the PD network



A higher value of "`auprc-2` - `auprc-0`" means that protein-protein link contained more information than protein node information for side effect prediction.
