import os
import bson
import numpy as np 
import pandas as pd

'''
This script gets the image ID, category ID and number of images per ID. 
It is used by the submission maker.

Because some image ID have multiple images, the predicted category
for the ID is the category with the highest confidence. The submission
maker needs to know how many pictures to consider per ID.
'''


def get_meta_from_bson(basepath):

	# get train metadata
	data = bson.decode_file_iter(open(os.path.join(basepath, 'train.bson'), 'rb'))
	d = dict()
	for c, json in enumerate(data):
	    product_id = json['_id']
	    d[product_id] = {'category_id': json['category_id'], 'n_imgs': len(json['imgs'])}

	# save to csv
	d = pd.DataFrame.from_dict(json, orient='index')
	d.index.name = '_id'
	d.rename(columns={0: 'category_id', 1: 'num_imgs'}, inplace=True)
	d.to_csv(os.path.join(basepath, 'train_images.csv'))

	# get test metadata
	data = bson.decode_file_iter(open(os.path.join(basepath, 'test.bson'), 'rb'))
	d = dict()
	for c, json in enumerate(data):
	    d[json['_id']] = {'n_imgs': len(json['imgs'])}

	# save to csv
	d = pd.DataFrame.from_dict(d, orient='index')
	d.index.name = '_id'
	d.rename(columns={0: 'n_imgs'}, inplace=True)
	d.to_csv(os.path.join(basepath, 'test_images.csv'))


if __name__ == '__main__':
	BASE_PATH = '../data/'
	get_meta_from_bson(BASE_PATH)
