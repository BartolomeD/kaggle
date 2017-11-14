from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from models.resnet import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

class CdiscountTestset():
    ''' Cdiscount Testset
    
    The general architecture of this class is 
    meant to work with the PyTorch DataLoader.
    '''
    def __init__(self, root_dir, transform=None):
        self.images = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)
        img_id = self.images[idx]  
        if self.transform:
            image = self.transform(image)            
        return image, img_id

# define pre-processing pipeline
data_transform = transforms.Compose([
    transforms.Scale(130),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406],
                         std=[.229, .224, .225])
])

# create a dataloader
data = CdiscountTestset(root_dir='../data/test/', transform=data_transform)
test_loader = DataLoader(data, batch_size=50, shuffle=True, num_workers=1, pin_memory=False)

# load model and pretrained weights
model = resnet50(pretrained=False)
model.fc = nn.Linear(2048, 5270)
model.load_state_dict(torch.load('resnet50_3-epoch_finetune-fc.pth'))
model.cuda()
model.eval()

# make the predictions
data = np.array(['', '', ''])
for x, _id in tqdm(test_loader):
    x = Variable(x.cuda())
    conf, pred = model(x).max(1)
    
    # to numpy and concatenate
    data = np.vstack([data, np.hstack([np.array(_id).reshape(-1, 1), 
                                       pred.data.cpu().numpy(), 
                                       conf.data.cpu().numpy()])])
    
pd.DataFrame(data).to_csv('submission.csv', index=False)
