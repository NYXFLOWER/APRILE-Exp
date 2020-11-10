import numpy as np
import pandas as pd
import os
import pickle

# ############## For tree map #####################
se_map = pd.read_csv('./index-map/side-effect-map.csv', usecols=[1]).to_numpy().flatten()
se_go = dict()
go_p = dict()
for se_idx in range(861):
    file = './out/all-all-{}-2.0-0.99/name_space.csv'.format(se_idx)
    if os.path.exists(file):
        description = pd.read_csv(file, usecols=[2, 3, 6]).to_numpy().T
        description = description[1:, description[0] == 'biological_process'].T
        # update gene in go
        for g, p in description:
            if go_p.get(g) is None:
                go_p[g] = set()
            go_p[g].add(p)
        # update go associated with side effect
        se_go[se_map[se_idx]] = set(description[:, 0])
with open('./out/tree-map/se-go.pkl', 'bw') as f:
    pickle.dump(se_go, f)
with open('./out/tree-map/go-p.pkl', 'bw') as f:
    pickle.dump(go_p, f)


with open('./out/tree-map/se-go.pkl', 'br') as f:
    se_go = pickle.load(f)
with open('./out/tree-map/go-p.pkl', 'br') as f:
    go_p = pickle.load(f)

se, go = [], []
for s, g in se_go.copy().items():
    if g:
        se.append(s)
        go.append(g)
    else:
        se_go.pop(s)
goo = list(go_p.keys())
n = len(se)
go_map = {k:v for v, k in enumerate(goo)}
se_map = {k:v for v, k in enumerate(se)}
data = np.zeros(shape=(n, len(goo)), dtype=np.int)
for i in range(n):
    s = se_map[se[i]]
    for g in go[i]:
        data[s, go_map[g]] = 1
data = pd.DataFrame(data, columns=goo, index=se)


# jd = np.zeros(shape=(n, n), dtype=np.float)
# for i in range(n):
#     for j in range(n):
#         a, b = len(go[i] | go[j]), len(go[i] & go[j])
#         jd[i, j] = (a - b)/a
# data = pd.DataFrame(jd, columns=se)
# data.to_csv('./out/tree-map/data_se_go.csv', index=True)

labels = pd.read_csv('./out/tree-map/tmp.csv', usecols=[1]).to_numpy().T.flatten()
se_go_list = []
for i in range(n):
    for g in go[i]:
        se_go_list.append([se[i], g, labels[i]])
data = pd.DataFrame(se_go_list, columns=['side effect', 'GO', 'label'])
data.to_csv('./out/tree-map/list.csv', index=True)


# se, go, data = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
#
# for se_idx in range(861):
#     file = './out/all-all-{}-2.0-0.99/name_space.csv'.format(se_idx)
#     if os.path.exists(file):
#         go_idx = pd.read_csv(file, usecols=[3]).to_numpy().flatten()
#         go_idx = np.unique([int(i[3:]) for i in go_idx])
#         l = len(go_idx)
#
#         se = np.concatenate((se, [se_idx] * l))
#         go = np.concatenate((go, go_idx))
#         data = np.concatenate((data, [1/l]*l))
#
# coo = sp.coo_matrix((data, (se, go)), shape=(861, max(go)+1))
# sp.save_npz('./out/coo', coo)
# print(coo.toarray())

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
#
# coo = sp.load_npz('./out/coo.npz').toarray()
# import matplotlib.pyplot as plt
# # coo = (coo*10).astype(np.int)
# plt.imshow(coo)
# plt.savefig("./fig.png")

#
#


# se_array = np.where(coo.sum(axis=1) > 0)[0]         # 395 side effects
# go_array = np.where(coo.sum(axis=0) > 0)[0]         # 415 GO processes
#
# coo = coo[se_array, :]
# coo = coo[:, go_array]
#
# tmp = np.eye(coo.shape[1], dtype=int)
# aaa = np.concatenate((tmp, coo), axis=0)            # go + se
#
# # pca = PCA(n_components=2)
# # comp = pca.fit_transform(aaa)
#
# from sklearn.manifold import TSNE
# comp = TSNE(n_components=2).fit_transform(aaa)*20
#
# coo = sp.load_npz('./out/coo.npz')
#
# for sss in se_array:
#     plt.figure(figsize=(15, 10))
#     tmp = np.where(coo.row == sss)
#     for i, j in zip(coo.row[tmp], coo.col[tmp]):
#         se_i = np.where(se_array == i)[0][0] + 415
#         go_j = np.where(go_array == j)[0][0]
#         plt.plot([comp[se_i][0], comp[go_j][0]], [comp[se_i][1], comp[go_j][1]],
#                  color='gray')
#     plt.scatter(comp[:415, 0], comp[:415, 1], label='GO', marker='*')
#     plt.scatter(comp[415:, 0], comp[415:, 1], label='Side Effect', alpha=0.3)
#     plt.title('Side Effect - {}'.format(sss))
#     plt.legend()
#     plt.savefig('/Users/nyxfer/Docu/fig-se/side-effect-{}.png'.format(sss))
#
#
# for sss in go_array:
#     plt.figure(figsize=(15, 10))
#     tmp = np.where(coo.col == sss)
#     for i, j in zip(coo.row[tmp], coo.col[tmp]):
#         se_i = np.where(se_array == i)[0][0] + 415
#         go_j = np.where(go_array == j)[0][0]
#         plt.plot([comp[se_i][0], comp[go_j][0]], [comp[se_i][1], comp[go_j][1]],
#                  color='gray')
#     plt.scatter(comp[:415, 0], comp[:415, 1], label='GO', marker='*')
#     plt.scatter(comp[415:, 0], comp[415:, 1], label='Side Effect', alpha=0.3)
#     plt.title('GO - {}'.format(sss))
#     plt.legend()
#     # plt.show()
#     plt.savefig('/Users/nyxfer/Docu/fig-go/go-{}.png'.format(sss))
#
#
#
#
import scipy.sparse as sp
import numpy as np
coo = sp.load_npz('./out/coo.npz').toarray()
import matplotlib.pyplot as plt

se_array = np.where(coo.sum(axis=1) > 0)[0]         # 395 side effects
go_array = np.where(coo.sum(axis=0) > 0)[0]         # 415 GO processes

coo = coo[se_array, :]
coo = coo[:, go_array]

plt.figure(figsize=(40, 4))
plt.stem(range(415), coo.sum(axis=0), use_line_collection=True)
plt.title('Go Process Importance')
plt.show()
#
tmp = coo.copy()
tmp[tmp>0] = 1

plt.figure(figsize=(40, 4))
fig, ax = plt.subplots(figsize=(40, 4))
ax.stem(go_array, tmp.sum(axis=0), use_line_collection=True)
# ax.set_xticks(np.array(go_array.tolist()))
ax.set_title('Go Process Occurrence Frequency')
plt.show()
#
# a = pd.DataFrame(tmp, columns=go_array, index=se_array)
# a.to_csv('./data/se-go.csv')

