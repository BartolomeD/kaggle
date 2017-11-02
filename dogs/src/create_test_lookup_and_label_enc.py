import os
import pandas as pd
import pickle as pkl
from sklearn import preprocessing

# create lookup table for test data
test_imgs = os.listdir('../data/test/')
test_data = pd.DataFrame({'file': test_imgs, 'id': list(map(lambda x: x[:-4], test_imgs))})
test_data.to_csv('../data/test_data.csv', index=False)

# encode all trainset labels
labels = pd.read_csv('../data/labels.csv')
le = preprocessing.LabelEncoder()
le.fit(list(set(labels.breed.values)))
with open('../data/le.pkl', 'wb') as f:
    pkl.dump(le, f)
    