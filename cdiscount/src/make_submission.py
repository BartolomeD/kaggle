import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from models.resnet import *
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# load model
model = resnet50(pretrained=False)
model.fc = nn.Linear(2048, 5270)
model.load_state_dict(torch.load('resnet50_1-epoch_finetune-fc-lyr3-lyr4.pth'))
model.cuda()
model.eval()

# define preprocessing pipeline
transform = transforms.Compose([
    transforms.Scale(130),
    transforms.ToTensor(),
    lambda x: x.view(1,3,130,130)
    ])

# load labelencoder
with open('../data/labelencoder.pkl', 'rb') as f:
    labelencoder = pickle.load(f)

# load test metadata
test_meta = pd.read_csv('../data/test_images.csv')
out_dict = {}

# iterate over images
for img_id, num_imgs in tqdm(test_meta.values):
    preds = np.array([])
    confs = np.array([])

    # if multiple images per id, do vote predict
    for v in range(num_imgs):

        # load and process image
        image = Image.open('../data/test/{}-{}.jpg'.format(img_id, v))
        image = Variable(transform(image)).cuda()

        # get predictions, append
        conf, pred = model(image).max(1)

        confs = np.append(confs, conf.data.cpu().numpy())
        preds = np.append(preds, pred.data.cpu().numpy())   

    # vote predict (highest confidence class), append
    out_dict.update({img_id: labelencoder.inverse_transform(preds.astype(np.int)[int(np.argmax(confs))])})

# to dataframe, csv
results = pd.DataFrame.from_dict(out_dict, 'index')
results.index.name = '_id'
results.columns = ['category_id']
results.to_csv('../output/submission_resnet1ep_finetune-fc-3-4_votepred.csv')

print('\n Finished.')
