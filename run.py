from src.layers import *
from src.utils import visualize_graph
from itertools import product
import argparse
import textwrap
import pickle
import os

# -------------- parse input parameters --------------
parser = argparse.ArgumentParser(
    prog='PoSePath',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="A package for detecting polypharmacy side effects pathway",
    epilog=textwrap.dedent("""\
        Examples:
          - for the drug pair (D-88, D-95) and the side effect (SE-846)
                $ python run.py 88 95 846 1
          - for all drug pairs causing the side effect (SE-846 and SE-848)
                $ python run.py * * 846,848 1
          - for all side effects caused by the drug pairs (D-2, D-95), (D-2, D-107), (D-88, D-95), (D-88, D-107)
                $ python run.py 2,88 95,107 2, * 1
          - for the side effects (SE-846 and SE-848) caused by all drug pairs which include the durg (D-88)
                $ python run.py 88 * * 1
        """),
)
# //TODO: add combination mods
parser.add_argument("drug_index_1", type=str,
                    help="[int/int_list/*] from 0 to 283")
parser.add_argument("drug_index_2", type=str,
                    help="[int/int_list/*] from 0 to 283")
parser.add_argument("side_effect_index", type=str,
                    help="[int/int_list/*] from 0 to 1152")
parser.add_argument("regul_sore", type=float, default=1,
                    help="[float] higher sore -> smaller pp-subgraph")
parser.add_argument("-v", "--version", action='version', version='%(prog)s 1.0')
parser.add_argument('-s', action='store_true', default=False,
                    help="additional [flag] for training edges separately")

args = parser.parse_args('88,57 1,95,55 28 1'.split())


def args_parse(in_str):
    if not isinstance(in_str, str):
        print('the input arg parameter is not a string')
        return
    out = in_str.split(',')
    return [int(i) for i in out]


args.drug_index_1 = args_parse(args.drug_index_1)
args.drug_index_2 = args_parse(args.drug_index_2)
args.side_effect_index = args_parse(args.side_effect_index)

# -------------- set working directory --------------
root = os.path.abspath(os.getcwd())
data_dir = root + '/data/'
out_dir = root + '/out/'

out_pkl_dir, out_fig_dir = out_dir + "pkl", out_dir + "fig"
if not os.path.exists(out_pkl_dir):
    os.makedirs(out_pkl_dir)
if not os.path.exists(out_fig_dir):
    os.makedirs(out_fig_dir)


# -------------- load data --------------
with open(data_dir + 'tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)


# -------------- parse drug pairs and side effects --------------
# //TODO: change args from int to string (list of int)
s = 28
p = 0.98
args.regul_score = 2.0
r = data.train_range[s]
tmp = data.train_idx[:, r[0]: r[1]]


# drug1, drug2 = tmp[0].tolist(), tmp[1].tolist()
drug1, drug2 = [91], [84]
side_effect = [s for i in range(len(drug1))]



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


# -------------- edge filter --------------
data = data.to(device)
model = model.to(device)
model.eval()
pp_static_edge_weights = torch.ones((data.pp_index.shape[1])).to(device)
pd_static_edge_weights = torch.ones((data.pd_index.shape[1])).to(device)
z = model.pp(data.p_feat, data.pp_index, pp_static_edge_weights)
z = model.pd(z, data.pd_index, pd_static_edge_weights)
P = torch.sigmoid((z[drug1] * z[drug2] * model.mip.weight[side_effect]).sum(dim=1))
# print(P.tolist())

drug1 = torch.Tensor(drug1)[P > p].tolist()
drug2 = torch.Tensor(drug2)[P > p].tolist()
side_effect = [s for i in range(len(drug2))]

# -------------- load and train explainer --------------
exp = Tip_explainer(model, data, device)
# //TODO: the input of explain() has been changed
result = exp.explain(drug1, drug2, side_effect, regulization=args.regul_sore)
# print(result)

# -------------- visualize p-p subgraph and save with png format --------------
# //TODO: rewrite the draw api for two mod.....
pp_idx, pp_weight, pd_idx, pd_weight = result
fig_name = '-'.join([str(drug1), str(drug2), str(side_effect), str(args.regul_sore)])

# visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, data.pp_index,
#                 out_fig_dir+"/{}.png".format(fig_name),
#                 protein_name_dict=data.prot_idx_to_id,
#                 drug_name_dict=data.drug_idx_to_id)
visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, data.pp_index, drug1, drug2,
                out_fig_dir+"/{}.png".format(fig_name), hiden=False)       # //TODO

# -------------- save results as dictionary in a pickle file --------------
out = {"pp_idx": pp_idx,
       "pp_weight": pp_weight,
       "pd_idx": pd_idx,
       "pd_weight": pd_weight}
with open(out_pkl_dir + "/{}.pkl".format(fig_name), "wb") as f:
    pickle.dump(out, f)

print(drug1)
print(drug2)