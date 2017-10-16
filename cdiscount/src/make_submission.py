import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from resnet import *

import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

# load model
model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, 5270)
model.load_state_dict(torch.load('resnet50_1ep_finetuneClf.pth'))
model.cuda()
model.eval()

# define preprocessing pipeline
transform = transforms.Compose([
    transforms.Scale(160),
    transforms.ToTensor(),
    lambda x: x.view(1,3,160,160)
    ])

# load labelencoder
with open('../data/labelencoder.pkl', 'rb') as f:
    labelencoder = pickle.load(f)

# load test metadata
test_meta = pd.read_csv('../data/test_images.csv')
preds = {}

# iterate over images
for img_id, num_imgs in tqdm(test_meta.values):
    classes = []
    confs = []

    # if multiple images per id, do vote predict
    for v in range(num_imgs):

        # load and process image
        image = Image.open('../data/test/{}-{}.jpg'.format(img_id, v))
        image = Variable(transform(image)).cuda()

        # get predictions, append
        conf, pred = model(image).max(1)
        confs.append(float(conf.data.numpy()))
        classes.append(int(pred.data.numpy()))

    # vote predict (highest confidence class), append
    preds.update({img_id: labelencoder.inverse_transform(classes[np.argmax(confs)])})

# to dataframe, csv
results = pd.DataFrame.from_dict(preds, columns=['_id','category_id'])
results.to_csv('../output/submission_resnet2ep_finetuneClf_votepred.csv', index=False)

print('\n Finished.')
