import pandas as pd
import numpy as np
import urllib

x = pd.read_csv('../data/labels.csv')
breeds = np.array(list(set(x.breed.values)))

with open('../data/wnid.txt', 'r') as f:
    d = f.readlines()
    wnid = [q.split(' ')[0] for q in d]
    label = [q.split(' ')[2][:-1].lower() for q in d]
    
x = list(map(lambda x: (True, x) if x in breeds else (False, x), label))
breedes = np.array([l[1] for l in x])[[l[0] for l in x]]

df = pd.DataFrame(np.vstack([np.array(wnid)[[l[0] for l in x]], breedes])).T
df.columns = ['wnid', 'breed']
df = df.drop_duplicates(subset='breed')

url = 'http://www.image-net.org/download/synset?wnid={}&username=danielbartolo1&accesskey=2a5ebf8cce24a09b112171a5ad1ac3023843d11d&release=latest&src=stanford'

for i, wnid in enumerate(df.wnid.values):
    print('\rDownloading {} [{}/{}]'.format(wnid, i+1, len(df)), end='')
    dl_url = url.format(wnid)
    urllib.request.urlretrieve(dl_url, '../data/imagenet/'+wnid+'.tar')
