from src.layers import *
from src.utils import visualize_graph
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
parser.add_argument("drug_index_1", type=str, default='88',
                    help="[int/int_list/*] from 0 to 283")
parser.add_argument("drug_index_2", type=str, default='95',
                    help="[int/int_list/*] from 0 to 283")
parser.add_argument("side_effect_index", type=str, default='846',
                    help="[int/int_list/*] from 0 to 1152")
parser.add_argument("regul_sore", type=float, default=1.0,
                    help="[float] higher sore -> smaller pp-subgraph")
parser.add_argument("-v", "--version", action='version', version='%(prog)s 1.0')
parser.print_help()

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
# //TODO: change args from int to string (list of int)
drug1, drug2, side_effect = args.drug_index_1, args.drug_index_2, args.side_effect_index
# //TODO: the input of explain() has been changed
result = exp.explain(int(drug1), int(drug2), int(side_effect), regulization=args.regul_sore)


# -------------- visualize p-p subgraph and save with png format --------------
# //TODO: rewrite the draw api for two mod.....
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
