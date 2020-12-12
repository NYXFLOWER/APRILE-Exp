import pubchempy as pcp
import pickle
import pandas as pd

with open("./data/tipexp_data.pkl", 'rb') as f:
    drug_map = pickle.load(f)
    drug_map = drug_map.drug_id_to_idx

drug_class = ['Covid-19',
              'Annaesthetics and Muscle Relaxants',
              'Analgesics',
              'Anti-arrhythmics',
              'Antibacterials',
              'Anti-coagulant, Anti-platelet and Fibrinolytic',
              'Anti-convulsants',
              'Anti-depressannts',
              'Anti-diabetics',
              'Antifungals',
              'Antipsychotics-Neuroleptics',
              'Anxiolytics-Hypnotics-sedatives',
              'Beta Blockers',
              'Bronchodilators',
              'Calcium Channel Blockers',
              'Conteraceptives-HRT',
              'Gastro-intestinal Agents (Anti-emetics)',
              'HCV DAAS',
              'HIV Antiretroviral Therapies',
              'Hypertension-Heart Failure Agents',
              'Immuno-suppressannts',
              'Inotropes and Vasopressors',
              'Lipid Lowering Agents',
              'Other',
              'Steroids']


out_table_columns = ['CID', 'PoSEPath Index', 'Drug Name', 'Drug Class']
out_table = []

for dc in drug_class:
    with open("./eva/cov19/raw/{}.drug".format(dc), 'r') as f:
        content = f.readlines()
        content = content[0]
    out_content = content[0]
    for s in content[1:]:
        if s.isupper() or s == '/':
            out_content += '\n'
        if s == '/' or s == '(' or s == ')':
            out_content += '\n'
            continue
        out_content += s
    out_content = out_content.split('\n')

    for s in out_content:
        if s:
            if len(s) < 2:
                continue
            c = pcp.get_compounds(s, 'name')
            if c:
                cid = c[0].cid
                print(s, cid)
                if drug_map.get('CID' + str(cid)) is not None:
                    out_table.append([cid, drug_map['CID' + str(cid)], s, dc])

out_df = pd.DataFrame(out_table, columns=out_table_columns)
out_df.to_csv('./eva/cov19/overlapping.tsv', sep='\t', index=False)
