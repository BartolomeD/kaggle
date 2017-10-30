import pickle
import bson
import pandas as pd
from sklearn.preprocessing import LabelEncoder

'''
Creates a labelencoder for id <=> class lookup and a categories.csv used by bson_to_jpg.py
'''

documents = bson.decode_file_iter(open('../data/train.bson', 'rb'))

# create (id <-> class) lookup table
categories = [(d['_id'], d['category_id']) for d in documents]
categories = pd.DataFrame(categories, columns=['id', 'cat'])
categories.to_csv('../data/categories.csv')

# create labelencoder file
labelencoder = LabelEncoder()
labelencoder.fit(categories.cat.unique().ravel())
with open('../data/labelencoder.pkl', 'wb') as f:
    pkl.dump(labelencoder, f)