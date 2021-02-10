from __future__ import print_function
from pubchempy import Compound

from goatools.base import \
    download_go_basic_obo  # Get http://geneontology.org/ontology/go-basic.obo
from goatools.base import \
    download_ncbi_associations  # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.test_data.genes_NCBI_9606_ProteinCoding import GENEID2NT as GeneID2nt_hum
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.godag_plot import plot_gos, plot_results, plot_goid2goobj

from src.utils import visualize_graph, args_parse_train, args_parse_pred
from src.layers import *
import pandas
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
          - for the drug pair (D-88, D-95) and the side effect (SE-846) if prediction prob more than 0.7
                $ python run.py 88 95 846 1 0.7
          - for all drug pairs causing the side effect (SE-846 and SE-848) (case: Prob > 0.99)
                $ python run.py all all 846,848 2 0.99
          - for all side effects caused by the drug pairs (D-2, D-95), (D-2, D-107), (D-88, D-95), (D-88, D-107)
                $ python run.py 2,88 95,107 2, all 2 0.89
          - for the side effects (SE-846 and SE-848) caused by all drug pairs which include the durg (D-88)
                $ python run.py 88 all all 3 0.9
                
        Index Map:
            See detailed drug, protein, side effect information in `../index-map` directory
                
        Output:
          - exp_info.pkl:                                       [dict] 
                1. drug1, drug2, side_effect: list of int
                2. pp_idx, pd_idx: int ndarray (2, n_edge), undirected
                3. pp_weight, pd_weight: float nparray (n_edges,), (0, 1]
                4, probability: list of float (0, 1)
          - visual_explainer.png:                               [figure]
                visualise pp_idx, pd_idx, pp_weight, pd_weight, drug1, drug2
          - go_enrich.png, go_enrich_symbols                    [figure]
                visualise significant GO terms with/without gene symbols
          - namespace.csv                                       [table]
                associated significant GO terms' name, namespace, p-value
        """))
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
parser.add_argument('-s', '--seed', default=0,
                    help="[int] random seed for optimizer")
parser.add_argument('-a', '--ifaddition', action='store_true', default=False,
                    help='add this flag for draw additional interactions')
parser.add_argument("-v", "--version", action='version', version='%(prog)s 3.0')

args = parser.parse_args()

if args.seed:
    torch.manual_seed(args.seed)
    print("==== SEED: {} ====".format(args.seed))
# -------------- set working directory --------------
root = os.path.abspath(os.getcwd())
data_dir = root + '/data/'

# -------------- load data --------------
with open(data_dir + 'tipexp_data.pkl', 'rb') as f:
    data = pickle.load(f)
# with open(data_dir + 'network.pkl', 'rb') as f:
#     network = pickle.load(f)

# -------------- parse drug pairs and side effects --------------
# //TODO: change args from int to string (list of int)
# drug1, drug2, side_effect = args_parse_train(args.drug_index_1, args.drug_index_2,
#                                              args.side_effect_index, data.train_range,
#                                              data.train_et, data.train_idx)
drug1, drug2, side_effect = args_parse_pred(args.drug_index_1, args.drug_index_2,
                                            args.side_effect_index,
                                            data.n_drug, data.n_et)

# -------------- identify device --------------
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print("==== DEVICE: {} ====".format(device_name))
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
P = torch.sigmoid(
    (z[drug1] * z[drug2] * model.mip.weight[side_effect]).sum(dim=1)).to("cpu")

# print(P[P>0.5].tolist())

tmp = P > args.filter
drug1 = torch.Tensor(drug1)[tmp].tolist()
if not drug1:
    raise ValueError("No Satisfied Edges." +
                     "\n - Suggestion: reduce the threshold probability with [-f] flag."
                     + "Current probability threshold is {}. ".format(
        args.filter) +
                     "\n - Suggestion: check if retrieved edge is in the training set."
                     "\n - Please use -h for help")

drug2 = torch.Tensor(drug2)[tmp].tolist()
side_effect = torch.Tensor(side_effect)[tmp].tolist()

# -------------- load and train explainer --------------
exp = PoseExplainer(model, data, device)
result = exp.explain(drug1, drug2, side_effect, regulization=args.regul_sore)
# print(result)

# -------------- visualize p-p subgraph and save with png format --------------
pp_idx, pp_weight, pd_idx, pd_weight = result
print('pp_edge: {}, pd_edge:{}, dd_edge:{}\n'.format(pp_idx.shape[1],
                                                     pd_idx.shape[1],
                                                     len(drug1)))

fig_name = '-'.join([args.drug_index_1, args.drug_index_2,
                     args.side_effect_index, str(args.regul_sore),
                     str(args.filter)])
if args.seed:
    fig_name += "-{}".format(args.seed)

if len(fig_name) > 30:
    print("The output files' name are 'tmp', as it is too lang.")
    fig_name = 'tmp'

out_dir = root + '/out_Covid/' + fig_name    # TODO
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print('==== OUTPUT DIR: {} ==='.format(out_dir))

G = visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, data.pp_index, drug1,
                    drug2,
                    out_dir + "/visual_explainer.png".format(fig_name),
                    hiden=args.ifaddition,
                    size=(60, 60),
                    protein_name_dict=data.prot_idx_to_id,
                    drug_name_dict=data.drug_idx_to_id)
print('SAVE -> visual explainer in visual_explainer.png')
with open(out_dir + "/graph.pkl", "wb") as f:
    pickle.dump(G, f)
    print('SAVE -> visual explainer in graph.pkl')

# -------------- save results as dictionary in a pickle file --------------
# print(pp_idx.device)
out = {"pp_idx": pp_idx,
       "pp_weight": pp_weight,
       "pd_idx": pd_idx,
       "pd_weight": pd_weight,
       "drug1": drug1,
       "drug2": drug2,
       "side_effect": side_effect,
       "probability": torch.Tensor(P[tmp]).tolist()}
with open(out_dir + "/exp_info.pkl", "wb") as f:
    pickle.dump(out, f)
print('SAVE -> trained explanation details in exp_info.pkl\n')

print('==== RETRIEVED INFO ====')
if len(drug1) < 15:
    print('      drug1:   ', drug1)
    print('      drug2:   ', drug2)
    print('  side effect: ', side_effect)
    print('  probability: ', ['%.2f' % i for i in torch.Tensor(P[tmp]).tolist()])
else:
    print('  Too many edges to print')
print()

info_table = pandas.DataFrame({'drug1': drug1,
                               'drug2': drug2,
                               'side effect': side_effect,
                               'probability': torch.Tensor(P[tmp]).tolist(),
                               'side effect name': [data.side_effect_idx_to_name[i] for i in np.array(side_effect).astype(np.int)]})
info_table.to_csv(out_dir+'/pred_results.csv', index=False)
print('SAVE -> predicted results in pred_results.csv\n')

# -------------- Go Enrichment --------------
print('==== GO ENRICHMENT ====')
obo_fname = download_go_basic_obo()
fin_gene2go = download_ncbi_associations()
obodag = GODag("go-basic.obo")

# Read NCBI's gene2go. Store annotations in a list of namedtuples
objanno = Gene2GoReader(fin_gene2go, taxids=[9606])

# Get namespace2association where:
#    namespace is:
#        BP: biological_process
#        MF: molecular_function
#        CC: cellular_component
#    association is a dict:
#        key: NCBI GeneID
#        value: A set of GO IDs associated with that gene
ns2assoc = objanno.get_ns2assc()

for nspc, id2gos in ns2assoc.items():
    print("{NS} {N:,} annotated human genes".format(NS=nspc, N=len(id2gos)))

goeaobj = GOEnrichmentStudyNS(
    GeneID2nt_hum.keys(),  # List of human protein-acoding genes
    ns2assoc,  # geneID/GO associations
    obodag,  # Ontologies
    propagate_counts=False,
    alpha=0.05,  # default significance cut-off
    methods=['fdr_bh'])  # default multipletest correction method

# 'p_' means "p-value". 'fdr_bh' is the multipletest method we are currently using.
geneids_study = pp_idx.flatten()  # geneid2symbol.keys()
geneids_study = [int(data.prot_idx_to_id[idx].replace('GeneID', ''))
                 for idx in geneids_study]

# ##############################################################################
goea_results_all = goeaobj.run_study(geneids_study)
goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
# ##############################################################################

n_sig = len(goea_results_sig)
print('\n==== FIND: {} significant GO terms ===='.format(n_sig))
if n_sig == 0:
    exit()

print()
# -------------- analysis --------------
geneid2symbol = {v.GeneID: v.Symbol for k, v in GeneID2nt_hum.items()}

keys = ['name', 'namespace', 'id']
df_go1 = pandas.DataFrame([{k: g.goterm.__dict__.get(k) for k in keys}
                           for g in goea_results_sig])
df_p = pandas.DataFrame([{'p_fdr_bh': g.__dict__['p_fdr_bh']}
                         for g in goea_results_sig])
df_go = df_go1.merge(df_p, left_index=True, right_index=True)

go_genes = pandas.DataFrame(
    [{'id': g.goterm.id, 'gene': s, 'symbol': geneid2symbol[s]}
     for g in goea_results_sig
     for s in g.study_items])
df_go = df_go.merge(go_genes, on='id')
print('SAVE -> name spaces of the significant GO terms in namespace.csv')
print(df_go.groupby('namespace').count())
df_go.to_csv(out_dir + "/name_space.csv")

print()
# -------------- Plot subset starting from these significant GO terms --------------
goid_subset = [g.GO for g in goea_results_sig]
plot_gos(out_dir + "/go_enrich.png",
         goid_subset,  # Source GO ids
         obodag,
         goea_results=goea_results_all)  # Use pvals for coloring

plot_gos(out_dir + "/go_enrich_symbols.pdf",
         goid_subset,  # Source GO ids
         obodag,
         goea_results=goea_results_all,  # use pvals for coloring
         id2symbol=geneid2symbol,
         # Print study gene Symbols, not Entrez GeneIDs
         study_items=6,  # Only only 6 gene Symbols max on GO terms
         items_p_line=3)  # Print 3 genes per line

print()
cids = [int(data.drug_idx_to_id[c].replace('CID', '')) for c in pd_idx[1]]
drugs = [Compound.from_cid(int(cid)) for cid in set(cids)]
drug_ids = pandas.DataFrame([[data.drug_id_to_idx['CID{}'.format(d.cid)],
                              d.cid,
                              'NA' if len(d.synonyms) == 0 else d.synonyms[0],
                              d.iupac_name] for d in drugs],
                            columns=["drug_idx", 'CID', 'synonym',
                                     'iupac_name'])
drug_ids.to_csv(out_dir + '/drug_details.csv', index=False)
print('SAVE -> drug compound info to drug_details.csv')
if drug_ids.shape[0] < 5:
    print(drug_ids)
print()

# -------------- Protein Dropout --------------
model.eval()
n = int(len(out['drug1']) / 2) if len(out['drug1']) > 2 else len(out['drug1'])
unique_p = set(out['pp_idx'].flatten()) | set(out['pd_idx'][0])
dropout_table = pandas.DataFrame({'drug_1': [int(d) for d in out['drug1'][:n]],
                                  'drug_2': [int(d) for d in out['drug2'][:n]],
                                  'side effect': [int(d) for d in
                                                  out['side_effect'][:n]]})
dropout_mask = dropout_table.copy()
dropout_mask['all'] = 0

pp_weights = torch.tensor([0.0001] * out['pp_idx'].shape[1]).to(device)
pd_weights = torch.tensor([0.0001] * out['pd_idx'].shape[1]).to(device)
pp_index = torch.tensor(out['pp_idx']).to(device)
pd_index = torch.tensor(out['pd_idx']).to(device)

z = model.pp(data.p_feat, pp_index, pp_weights)
z = model.pd(z, pd_index, pd_weights)
P = torch.sigmoid(
    (z[drug1] * z[drug2] * model.mip.weight[side_effect]).sum(dim=1)).to("cpu")
dropout_table['all'] = P.tolist()[:n]

for p in unique_p:
    pp_weights = np.copy(out['pp_weight'])
    pp_weights[out['pp_idx'][0] == p] = 0.0001
    pp_weights[out['pp_idx'][1] == p] = 0.0001
    pp_weights = torch.tensor(pp_weights).to(device)
    z = model.pp(data.p_feat, pp_index, pp_weights)

    pd_weights = np.copy(out['pd_weight'])
    pd_weights[out['pd_idx'][0] == p] = 0.0001
    pd_weights = torch.tensor(pd_weights).to(device)
    z = model.pd(z, pd_index, pd_weights)

    P = torch.sigmoid(
        (z[drug1] * z[drug2] * model.mip.weight[side_effect]).sum(dim=1)).to(
        "cpu")
    dropout_table['gene {}'.format(p)] = P.tolist()[:n]
    dropout_mask['gene {}'.format(p)] = 0
dropout_table.to_csv(out_dir + '/gene_dropout.csv', index=False)
print('SAVE -> gene dropout results to gene_dropout.csv')

pd = out['pd_idx']
for i in range(dropout_mask.shape[0]):
    d1, d2 = dropout_table.loc[i, 'drug_1'], dropout_table.loc[i, 'drug_2']
    ppp = pd[:, (pd[1] == d1) | (pd[1] == d2)]
    for p in ppp[0]:
        dropout_mask.loc[i, 'gene {}'.format(p)] = 1
dropout_mask.to_csv(out_dir + '/gene_dropout_mask.csv', index=False)
print('SAVE -> gene dropout results to gene_dropout_mask.csv')
