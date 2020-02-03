from src.layers import *
from src.utils import visualize_graph, args_parse
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
                $ python run.py all all 846,848 1
          - for all side effects caused by the drug pairs (D-2, D-95), (D-2, D-107), (D-88, D-95), (D-88, D-107)
                $ python run.py 2,88 95,107 2, all 1
          - for the side effects (SE-846 and SE-848) caused by all drug pairs which include the durg (D-88)
                $ python run.py 88 all all 1
        """),
)
# //TODO: add combination mods
parser.add_argument("drug_index_1", type=str,
                    help="[int/int_list/all] from 0 to 283")
parser.add_argument("drug_index_2", type=str,
                    help="[int/int_list/all] from 0 to 283")
parser.add_argument("side_effect_index", type=str,
                    help="[int/int_list/all] from 0 to 860")
parser.add_argument("regul_sore", type=float, default=1.0,
                    help="[float] higher sore -> smaller pp-subgraph")
parser.add_argument('filter', type=float, default=0.98,
                    help='[float] threshold probability')
parser.add_argument('-a', '--ifaddition', action='store_true', default=False,
                    help='add this flag for draw additional interactions')
parser.add_argument("-v", "--version", action='version', version='%(prog)s 2.0')
args = parser.parse_args()


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
drug1, drug2, side_effect = args_parse(args.drug_index_1, args.drug_index_2,
                                       args.side_effect_index, data.train_range,
                                       data.train_et, data.train_idx)


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


tmp = P > args.filter
drug1 = torch.Tensor(drug1)[tmp].tolist()
if not drug1:
    raise ValueError("No Satisfied Edges." +
                     "\n - Suggestion: reduce the threshold probability with [-f] flag."
                     + "Current probability threshold is {}. ".format(args.f) +
                     "\n - Suggestion: check if retrieved edge is in the training set."
                     "\n - Please use -h for help")

drug2 = torch.Tensor(drug2)[tmp].tolist()
side_effect = torch.Tensor(side_effect)[tmp].tolist()

# -------------- load and train explainer --------------
exp = Tip_explainer(model, data, device)
# //TODO: the input of explain() has been changed
result = exp.explain(drug1, drug2, side_effect, regulization=args.regul_sore)
# print(result)

# -------------- visualize p-p subgraph and save with png format --------------
# //TODO: rewrite the draw api for two mod.....
pp_idx, pp_weight, pd_idx, pd_weight = result

if len(drug1) > 15:
    fig_name = 'tmp'
else:
    fig_name = '-'.join([args.drug_index_1, args.drug_index_2,
                         args.side_effect_index, str(args.regul_sore),
                         str(args.filter)])

# visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, data.pp_index,
#                 out_fig_dir+"/{}.png".format(fig_name),
#                 protein_name_dict=data.prot_idx_to_id,
#                 drug_name_dict=data.drug_idx_to_id)
visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, data.pp_index, drug1, drug2,
                out_fig_dir+"/{}.png".format(fig_name),
                hiden=args.ifaddition,
                size=(60, 60),
                protein_name_dict=data.prot_idx_to_id,
                drug_name_dict=data.drug_idx_to_id)       # //TODO
print('figure is saved in ./out/fig')

# -------------- save results as dictionary in a pickle file --------------
print(pp_idx.device)
out = {"pp_idx": pp_idx,
       "pp_weight": pp_weight,
       "pd_idx": pd_idx,
       "pd_weight": pd_weight}
with open(out_pkl_dir + "/{}.pkl".format(fig_name), "wb") as f:
    pickle.dump(out, f)
print('result is saved in ./out/pkl')

if len(drug1) < 15:
    print(drug1)
    print(drug2)
    print(side_effect)
