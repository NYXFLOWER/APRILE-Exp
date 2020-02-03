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
