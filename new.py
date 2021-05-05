#%%
import os
# os.chdir('PoSePath')

from src.layers import *
import pandas as pd
from pubchempy import Compound

root = os.path.abspath(os.getcwd())
data_dir = os.path.join(root, 'data/')
out_dir = os.path.join(root, 'out_ui/')

pose = Pose(data_dir, device='cuda')
pred = pose.get_prediction_train(threshold=0.99)

columns=['drug1', 'drug2', 'side effect', 'prob', 'pui', 'ppui']
df = pd.DataFrame(pred).T
df.columns = columns
df = df.sort_values(by='prob', ascending=False)

d_list = ['drug1', 'drug2']
for dname in d_list:
    cids = [int(pose.data.drug_idx_to_id[c].replace('CID', '')) for c in df[dname]]
    drugs = [Compound.from_cid(int(cid)) for cid in cids]
    df[f'{dname} cid'] = [d.cid for d in drugs]
    df[f'{dname} name'] = [d.iupac_name for d in drugs] 

df = df[:20]

df['side effect name'] = [pose.data.side_effect_idx_to_name[int(i)] for i in df['side effect']]
df.to_csv('rank99.csv', index=False)
df.iloc[list(range(0, 20, 2)),:]

for d1, d2, se in df[['drug1', 'drug2', 'side effect']][:20].values:
    os.system(f"python run.py {int(d1)} {int(d2)} {int(se)} 2 0.5")
    print(d1, d2, se)

# test single prediction
# pred = pose.predict([11], [12], [0])
# print(pred)

# query = pose.explain([35], [152], [300])
# i = 0
# query.save(f'out_top_train/{i}.pkl')

print(EPS)