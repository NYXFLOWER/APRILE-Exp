In `SupData.xlsx`, there are 11 sheets:
1. `D_mapping`: drug index, PubChem identifier, synonyms, and IUPAC name
2. `P_mapping`: Protein index, GeneID, and gene symbol
3. `SE_mapping`: Side effect index and name
4. `SE_MeSH_category`: Side effect index and name, as well as MeSH category symbol and name
5. `SE_cluster`: Side effect index and cluster name
6. `SE-GO_adj_with_#gene`: Each row is a side effect, and each column is a GO item. The value in each cell is the number of genes identified by explainer for the side effect (row) in the GO(column).
7. `sepsis_hi`: Explanations for sepsis with high PPIU
8. `sepsis_low`: Explanations for sepsis with low PPIU
9. `cluster_enrich_mesh`: MeSH category, cluster center, P-value of over-representation of MeSH category based on Fisher's exact test, odds ratio from Fisher's exact test, cluster size, Benjamini-Hochberg adjusted P-value, and category name
10. `CTD_permutation_test`:
	- side effect index and name
	- permutation P-value for precision and Jaccard index
	- precision
	- Jaccard index
	- mean precision and Jaccard index of 10,000 random explanations
	- random model: nongo_strong (entire subgraph, not necessarily GO-enriched; preserve target and non-target numbers), nongo_weak (entire subgraph, not necessarily GO-enriched; not preserving target and non-target numbers), go_strong (subgraph with GO-enriched genes only, preserve target and non-target numbers), go_weak (subgraph wit GO-enriched genes only, not preserving target and non-target numbers)
	- number of predicted proteins, genes in CTD associated with the side effect (disease), target genes and non-target genes in the explanation, and random explanations
	- 100% * (Precision_predicted - Mean(Precision_random)) / Mean(Precision_random)
	- 100% * (Jaccard_predicted - Mean(Jaccard_random)) / Mean(Jaccard_random)
11. `cluster_enrich_GO`: Gene Ontology term, cluster center (exemplar), P-value of enrichment of GO term in cluster and odds ratio based on Fisher's exact test, cluster size and Benjamini-Hochberg adjusted P-value

Beside, the APRILE's explanations for the top 200 polypharmacy side effect predictions (ranked by predicted probability score) in training set are provided in `exp_top200_train.csv`.
