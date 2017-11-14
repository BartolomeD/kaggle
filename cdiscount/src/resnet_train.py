import os
import pickle
import io
import bson
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models.resnet import *
from torchvision import transforms

# writes loss to file
def write_loss(loss, acc):
    w = '{}, {}\n'.format(format(loss, '.3f'), format(acc, '.2f'))
    with open('exp4_loss.txt', 'a') as f:
        f.write(w)

def batch_generator(data_path, processing, batch_size=32, return_labels=True):
    
    # decode data
    data = bson.decode_file_iter(open(data_path, 'rb'))

    # iterate over data items
    x = torch.FloatTensor(()).cuda()
    y = torch.LongTensor(()).cuda()
    for item in data:

        # get item label
        category = item.get('category_id', '')
        label = int(labelencoder.transform([category])) if category else 0
        label = torch.LongTensor([label]).cuda()

        # get images in item and process
        for image in item.get('imgs'):

            # from binary, process, augment and to tensor
            proc_img = processing(Image.open(io.BytesIO(image.get('picture', None))))

            # add to batch
            x = torch.cat([x, proc_img])
            y = torch.cat([y, label])

            if x.size(0) == batch_size:

                if return_labels:
                    yield Variable(x), Variable(y)
                else:
                    yield Variable(x)
                
                x = torch.FloatTensor(()).cuda()
                y = torch.LongTensor(()).cuda()

def train(epoch):

    # init stats
    c = 0
    train_loss = 0
    train_acc = 0

    # set model to train mode and reset gradients
    model.train()
    optimizer.zero_grad()
    
    # iterate over training batches
    for batch_idx, (x, y) in enumerate(data_loader):

        # get batch predictions and loss
        output = model(x)
        loss = crit(output, y)
        
        # accumulate gradients
        loss.backward()
        if batch_idx % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        # accumulate statistics
        _, idx = output.cpu().max(1)
        train_loss += loss.data[0]
        train_acc += accuracy_score(y.cpu().data.numpy(), idx.data.numpy().ravel())
        c += 1

        # print statistics
        if batch_idx % print_iter == 0:

            # get average loss and accuracy
            train_loss /= c
            train_acc /= c

            # save loss and acc to file
            write_loss(train_loss, train_acc)

            # print the statistics
            print('\rEpoch {} [{}/{} ({:.0f}%)] - loss: {:.6f} - acc: {:.3f}'.format(
                epoch+1, batch_idx * batch_size, 12371293, 100. * batch_idx / (12371293//batch_size), 
                train_loss, train_acc), end='')
            
            # reset stats
            c = 0
            train_loss = 0
            train_acc = 0

        # exit training phase
        if batch_idx >= val_split:
            return

def test():

    # init stats
    test_loss = 0
    correct = 0

    # set model to evaluation mode
    model.eval()

    # iterate over validation batches
    for batch_idx, (x, y) in enumerate(data_loader):

        # forward pass plus stat accumulation
        output = model(x)
        test_loss += crit(output, y).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        if batch_idx % print_iter == 0:
            print('\rValidating [{}/{} ({:.0f}%)]'.format(
                val_split*batch_size+batch_idx*batch_size, 12371293, 
                100. * batch_idx / (12371293//batch_size)), end='')

    # print validation phase statistics
    test_loss /= (batch_idx + 1)
    print('\nValidation set - loss: {:.4f} - val-acc: {:.0f}%\n'.format(
        test_loss, (correct / ((batch_idx + 1) * batch_size))*100))

# load lookup table and labelencoder
with open('../data/labelencoder.pkl', 'rb') as f:
    labelencoder = pickle.load(f)

# parameters
batch_size = 32
image_size = 130
epochs = 1
num_classes = 5270
val_split = round(0.9*(12371293//batch_size))
accum_iter = 2
print_iter = 10

# preprocessing pipeline
processing = transforms.Compose([
    transforms.Scale(image_size),
    transforms.ToTensor(),
    lambda x: x.cuda().view(1,3,image_size,image_size),
    transforms.Normalize(mean=[.485, .456, .406],
                         std=[.229, .224, .225])
    ])

# load ResNet50 with ImageNet weights
model = resnet50(pretrained=False)

# freeze all parameters
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)

model.load_state_dict(torch.load('resnet50_2-epoch_finetune-fc.pth'))

# send model to GPU
model.cuda()

# # loss and optimizer
crit = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
#                       lr=1e-2, momentum=0.9, weight_decay=1e-6)

# # pre-train fc layer
# for e in range(1):
#     data_loader = batch_generator('../data/train.bson', processing, batch_size=batch_size)
#     train(e)
#     torch.save(model.state_dict(), './resnet50_{}-epoch_finetune-fc.pth'.format(e+1))
#     test()

# unfreeze fully-connected and 3rd/4th layer
for param in model.fc.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True

# update optimizer with new weights
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=1e-4, momentum=0.9, weight_decay=1e-6)

# train fc, layer3 and layer4
for e in range(epochs):
    data_loader = batch_generator('../data/train.bson', processing, batch_size=batch_size)
    train(e)
    torch.save(model.state_dict(), './resnet50_3-epoch_finetune-fc.pth')
    test()

print('\nFinished.')
