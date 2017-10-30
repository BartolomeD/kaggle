import os
import bson
import numpy as np
import pandas as pd
from tqdm import tqdm


def bson_to_jpg(basepath, train=False):
	'''Creates jpg data from source BSON file with classes as folder structure.'''
	out_folder = os.path.join(basepath, 'train') if train else os.path.join(basepath, 'test')

	# Create output folder
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	# Create categories folders
	if train:
		categories = pd.read_csv('category_names.csv', index_col='category_id')
		for category in tqdm(categories.index):
			os.mkdir(os.path.join(out_folder, str(category)))

	bar = tqdm(total = 7069896 if train else 1768182)
	with open(basepath+'test.bson', 'rb') as fbson:
		data = bson.decode_file_iter(fbson)
		for c, d in enumerate(data):
			_id = d['_id']
			_cat = str(d['category_id'])
			for e, pic in enumerate(d['imgs']):
				train_fname = os.path.join(out_folder, _cat, '{}-{}.jpg'.format(_id, e))       
				test_fname = os.path.join(out_folder, '{}-{}.jpg'.format(_id, e))
				fname = train_fname if train else test_fname
				with open(fname, 'wb') as f:
					f.write(pic['picture'])
			bar.update()


if __name__ == '__main__':
	BASE_PATH = 'path/to/data.bson'
	bson_to_jpg(BASE_PATH)


