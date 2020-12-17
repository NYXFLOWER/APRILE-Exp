import os
import sys
import pickle
import textwrap
import scholarly
import requests

# parse parameters
if sys.argv[1] in {'-h', '--help'}:
    print(textwrap.dedent("""\
    Usage: query [top_n] [query]
        [top_n]     int     the maximum number of matched papers to be returned
        [query]     str     keywords or paper titles or authors
        
    Example:
        python query.py 20 "arrhythmia" "SLCO1B1"
    """))
    exit()

top_n = int(sys.argv[1])
query = ' '.join(sys.argv[2:])
print("============== Query ==============\n", query)

# get top-n matched papers' bib (and link)
search_query = scholarly.search_pubs_query(query)
out = [next(search_query).bib for i in range(top_n)]

# text version
for i in range(top_n):
    if out[i].get("'abstract'"):
        out[i]["'abstract'"] = out[i]["'abstract'"].replace('\n', '')
out_str = '\n\n'.join(['Randed {}/{}\n'.format(i+1, top_n) + str(out[i]).replace("', '", "'\n'").replace('{', '').replace('}', '') for i in range(top_n)])

# out directory define
root = os.path.abspath(os.getcwd())
out_dir = root + '/query_result'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# save/write to files
with open(out_dir + '/{}.pkl'.format(query), 'wb') as f:
    pickle.dump(out, f)
with open(out_dir + '/{}.txt'.format(query), 'w') as f:
    f.writelines(out_str)

print()
print("============== Results ============== ")
print(out_str)
print("The results are saved in {}".format(out_dir))
