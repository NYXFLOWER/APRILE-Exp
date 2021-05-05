from src.layers import *
import os

class PoseExplainer():
    def __init__(level=NOTSET)

def load_model_and_data():
    root = os.path.abspath(os.getcwd())
    data_dir = root + '/data/'

    with open(data_dir + 'tipexp_data.pkl', 'rb') as f:
        data = pickle.load(f)

    nhids_gcn = [64, 32, 32]
    prot_out_dim = sum(nhids_gcn)
    drug_dim = 128
    pp = PP(data.n_prot, nhids_gcn)
    pd = PD(prot_out_dim, drug_dim, data.n_drug)
    mip = MultiInnerProductDecoder(drug_dim + pd.d_dim_feat, data.n_et)
    model = Model(pp, pd, mip).to('cpu')
    name = 'poly-' + str(nhids_gcn) + '-' + str(drug_dim)
    model.load_state_dict(torch.load(data_dir + name + '-model.pt'))
    model.eval()

    return model, data

