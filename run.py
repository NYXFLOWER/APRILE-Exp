import pickle
from src.layers import *
from src.utils import visualize_graph
import argparse
import os


# -------------- parse input parameters --------------
parser = argparse.ArgumentParser()
# parser.add_argument("root", type=str,
#                     help="str - the path of ")
parser.add_argument("drug_1", type=int,
                    help="int - drug index from 0 to 283")
parser.add_argument("drug_2", type=int,
                    help="int - drug index from 0 to 283")
parser.add_argument("side_effect", type=int,
                    help="int - side effect index from 0 to 1152")
parser.add_argument("regul_sore", type=float,
                    help="float - higher sore --> smaller pp-subgraph")
args = parser.parse_args()


# -------------- set working directory --------------
root = os.path.abspath(os.getcwd())
data_dir = root + '/data/'
out_dir = root + '/out/'

out_pkl_dir, out_fig_dir = out_dir + "/pkl", out_dir + "/fig"
if not os.path.exists(out_pkl_dir):
    os.makedirs(out_pkl_dir)
if not os.path.exists(out_fig_dir):
    os.makedirs(out_fig_dir)


# -------------- load data --------------
with open(data_dir + 'tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)


# -------------- identify device --------------
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device_name)
device = torch.device(device_name)


# -------------- load trained model --------------
nhids_gcn = [64, 32, 32]
prot_out_dim = sum(nhids_gcn)
drug_dim = 128
pp = PP(data.n_prot, nhids_gcn)
pd = PD(prot_out_dim, drug_dim, data.n_drug)
mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, data.n_et)
model = Model(pp, pd, mip).to('cpu')
name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)
model.load_state_dict(torch.load(data_dir + name + '-model.pt'))


# -------------- load explainer --------------
exp = Tip_explainer(model, data, device)


# -------------- given a drug pair and a side effect --------------
drug1, drug2, side_effect = args.drug_1, args.drug_2, args.side_effect
result = exp.explain(drug1, drug2, side_effect, regulization=args.regul_sore)


# -------------- visualize p-p subgraph and save with png format --------------
pp_idx, pp_weight, pd_idx, pd_weight = result
fig_name = '-'.join([str(drug1), str(drug2), str(side_effect), str(args.regul_sore)])

visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, data.pp_index,
                out_fig_dir+"/{}.png".format(fig_name),
                protein_name_dict=data.prot_idx_to_id,
                drug_name_dict=data.drug_idx_to_id)


# -------------- save results as dictionary in a pickle file --------------
out = {"pp_idx": pp_idx,
       "pp_weight": pp_weight,
       "pd_idx": pd_idx,
       "pd_weight": pd_weight}
with open(out_pkl_dir + "/{}.pkl".format(fig_name), "wb") as f:
    pickle.dump(out, f)
