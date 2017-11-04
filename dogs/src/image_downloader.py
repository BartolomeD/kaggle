import re
import os
import json
import urllib
import pandas as pd
from bs4 import BeautifulSoup

def download_images(breed, suffix):
    query = suffix+breed
    url = 'https://www.google.co.in/search?q={}&source=lnms&tbm=isch'.format(query)
    hdr = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36'}
    soup = BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url, headers=hdr)), 'html.parser')

    image_links = [json.loads(a.text)['ou'] for a in soup.find_all('div', {'class': 'rg_meta'})]
    print('{} images found'.format(len(image_links)))

    for i, img in enumerate(image_links):

        # filename = breed+,,,,,+url. 
        # gebruik .split(',,,,,')[0] op filename om class (breed) te krijgen
        # ik gebruik urls om unique namen te krijgen, en duplicates te voorkomen
        filename = breed+',,,,,'+re.sub('/', '-', img[10:])
        try:
            urllib.request.urlretrieve(img, '../data/new_imgs2/{}'.format(filename))
        except Exception as e:
            print('Could not load : {}'.format(img))
            print(e)


if __name__ == '__main__':        
    labels = pd.read_csv('../data/labels.csv')
    breeds = list(set(labels.breed.values))
    suffixes = ['dog+', 'sleeping+', 'running+', 'pup+']

    for breed in breeds:
        for suffix in suffixes:
            download_images(breed, suffix)
            