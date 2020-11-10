from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


adj_se_go = pd.read_csv("./data_se_go.csv", index_col=0)
se = adj_se_go.index
go = adj_se_go.columns
adj = adj_se_go.to_numpy()
plt.figure(figsize=(40, 4))
plt.stem(go, adj.sum(axis=0), use_line_collection=True)
plt.title('Go Process Importance')
plt.show()

# download and load GO and gene-GO dataset
obo_fname = download_go_basic_obo()
gene2go = download_ncbi_associations()
fin_gene2go = download_ncbi_associations()
objanno = Gene2GoReader(fin_gene2go, taxids=[9606])
ns2assoc = objanno.get_ns2assc()    # gene-GO association dataset
obodag = GODag("go-basic.obo")      # GO dataset

# sort GO array with side effect association from low to high
go_array = np.where(adj.sum(axis=0) > 90)
t = go[go_array]
sort_go = go[np.argsort(adj.sum(axis=0))]
sort_go_name = [obodag[i].name for i in sort_go]

# plot given GO item's association graph
from goatools.gosubdag.plot.plot import plot_results, plot_gos
plot_results("test_plot.png", [obodag[i] for i in sort_go], parentcnt=True, childcnt=True)
lst_go = list(sort_go)
plot_gos('testgo.pdf', list(sort_go), obodag)
plot_gos('111.pdf', [lst_go[-1], lst_go[-2]], obodag)


# key: GO; value: a set of gene in that GO -- biological process
go_gene_asso_bp = dict()
for k, v in ns2assoc['BP'].items():
    for gg in v:
        if go_gene_asso_bp.get(gg) is None:
            go_gene_asso_bp[gg] = set()
        go_gene_asso_bp[gg].add(k)

# key: GO; value: a set of gene in that GO -- molecular function
go_gene_asso_mf = dict()
for k, v in ns2assoc['MF'].items():
    for gg in v:
        if go_gene_asso_mf.get(gg) is None:
            go_gene_asso_mf[gg] = set()
        go_gene_asso_mf[gg].add(k)
