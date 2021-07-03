from src.layers import PoseQuery, gdata

def load_pred_exp_table(file_dir, top20):
    top20_query = []

    for n in range(top20.shape[0]):
        top_n = top20.iloc[n, :].values
        d1, d2, s = top_n[0], top_n[3], top_n[6]
        file_name = f"{file_dir}/{d1}-{d2}-{s}.pkl"
        top20_query.append(PoseQuery.load_from_pkl(file_name))

    target, ntarget, GOs = [], [], []
    for q in top20_query:
        ts = set(q.pd_index[0].tolist())
        nts = set(q.pp_index.flatten().tolist())
        gs =  nts | ts

        if len(nts):
            nts = [int(gdata.prot_idx_to_id[i][6:]) for i in gs - ts]
            nts = [gdata.geneid2symbol[i] if gdata.geneid2symbol.get(i) else 'NA' for i in nts]
            ntarget.append(','.join(nts))
        else:
            ntarget.append('NA') 

        if q.if_enrich:
            gos = q.get_GOEnrich_table()['name'].unique().tolist()
            GOs.append(','.join(gos))
        else:
            GOs.append('NA')

        ts = [int(gdata.prot_idx_to_id[i][6:]) for i in ts]
        ts = [gdata.geneid2symbol[i] if gdata.geneid2symbol.get(i) else 'NA' for i in ts]
        target.append(','.join(ts))

    top20['target'], top20['non-target'], top20['GOs'] = target, ntarget, GOs
    return top20
