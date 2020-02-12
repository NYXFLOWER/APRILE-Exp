from src.utils import visualize_graph
import numpy as np
import pickle
import os

n_p_node = 10
n_d_node = 10
n_pp_edges = 10
n_pd_edges = 5


pp_idx = np.random.randint(n_p_node, size=(2, n_pp_edges), )
pp_weight = np.random.random(size=(1, n_pp_edges))
pd_idx = np.random.randint(n_d_node, size=(2, n_pd_edges))
pd_weight = np.random.random(size=(1, n_pd_edges))

fig_name = "test"

# root = args.root
root = os.path.abspath(os.getcwd())
# root = '.'
src_dir = root + '/data/'
out_dir = root + '/out/'


with open(src_dir + 'tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)


visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, fig_name, data.pp_index,
                protein_name_dict=data.prot_idx_to_id,
                drug_name_dict=data.drug_idx_to_id)


out = {"pp_idx": pp_idx,
       "pp_weight": pp_weight,
       "pd_idx": pd_idx,
       "pd_weight": pd_weight}
directory = os.path.abspath(os.getcwd())+"/output/pkl"
if not os.path.exists(directory):
    os.makedirs(directory)
with open(directory + "/{}.pkl".format(fig_name), "wb") as f:
    pickle.dump(out, f)


#########################################
from pubchempy import Compound
import pandas as pd
cids = [int(data.drug_idx_to_id[i][3:]) for i in range(len(data.drug_idx_to_id))]
drugs = [Compound.from_cid(int(cid)) for cid in cids]
drug_ids = pd.DataFrame([[data.drug_id_to_idx['CID{}'.format(d.cid)], d.cid, 'NA' if len(d.synonyms) == 0 else d.synonyms[0], d.iupac_name] for d in drugs], columns=["drug_idx", 'CID', 'synonym', 'iupac_name'])

drug_ids.to_csv('./index-map/drug-map.csv', index=False)

from goatools.test_data.genes_NCBI_9606_ProteinCoding import GENEID2NT as GeneID2nt_hum
geneid2symbol = {v.GeneID: v.Symbol for k, v in GeneID2nt_hum.items()}
genes = [int(data.prot_idx_to_id[i][6:]) for i in range(len(data.prot_idx_to_id))]
gene_ids = pd.DataFrame([[data.prot_id_to_idx['GeneID{}'.format(gene)],
                          gene,
                          geneid2symbol[gene] if geneid2symbol.get(gene) else 'NA']
                         for gene in genes],
                        columns=['protein_idx', 'gene_id', 'gene_symbol'])
gene_ids.to_csv('./index-map/protein-map.csv', index=False)

se_ids = pd.DataFrame(data.side_effect_idx_to_name.items(), columns=['side_effect_idx', 'name'])
se_ids.to_csv('./index-map/side-effect-map.csv', index=False)