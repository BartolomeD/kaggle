import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.resnet import *
from data_loaders import train_valid_loader


def train(epoch, data_loader):

    # init stats
    c = 0
    train_loss = 0
    train_acc = 0

    # set model to train mode and reset gradients
    model.train()
    optimizer.zero_grad()
    
    # iterate over training batches
    for batch_idx, batch in enumerate(data_loader):

        inputs = Variable(batch['image']).cuda()
        targets = Variable(batch['label']).squeeze().cuda()

        # get batch predictions and loss
        outputs = model(inputs)
        loss = crit(outputs, targets)
        
        # accumulate gradients
        loss.backward()
        if batch_idx % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        # accumulate statistics
        _, idx = outputs.cpu().max(1)
        train_loss += loss.data[0]
        train_acc += accuracy_score(targets.cpu().data.numpy(), idx.data.numpy().ravel())
        c += 1

        # print statistics
        if batch_idx % print_iter == 0:

            # get average loss and accuracy
            train_loss /= c
            train_acc /= c

            # print the statistics
            print('\rEpoch {} [{}/{} ({:.0f}%)] - loss: {:.6f} - acc: {:.3f}'.format(
                epoch+1, batch_idx * batch_size, len(data_loader.dataset.labels), 
                100. * batch_idx / (len(data_loader.dataset.labels)//batch_size), 
                train_loss, train_acc), end='')
            
            # reset stats
            c = 0
            train_loss = 0
            train_acc = 0


def test(epoch, data_loader):

    # init stats
    test_loss = 0
    correct = 0

    # set model to evaluation mode
    model.eval()

    # iterate over validation batches
    for batch_idx, batch in enumerate(data_loader):

        inputs = Variable(batch['image']).cuda()
        targets = Variable(batch['label']).squeeze().cuda()

        # forward pass
        outputs = model(inputs)
        test_loss += crit(outputs, targets).data[0]

        # stat accumulation
        pred = outputs.data.max(1)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    # print validation phase statistics
    test_loss /= (batch_idx + 1)
    print('\nValidation set - loss: {:.4f} - val-acc: {:.0f}%\n'.format(
        test_loss, (correct / ((batch_idx + 1) * batch_size))*100))

# load lookup table and labelencoder
with open('../data/le.pkl', 'rb') as f:
    le = pickle.load(f)

# data parameters
data_dir = '../data/train/'
labels = '../data/labels.csv'
seed = 1337
batch_size = 32
image_size = 130
valid_split = 0.15
augment = True
shuffle = True

# model/training parameters
num_classes = len(le.classes_)
learning_rate = 1e-2
accum_iter = 1
num_epochs = 10
print_iter = 10

# load training phase loaders
train_loader, val_loader = train_valid_loader(data_dir=data_dir,
                                              labels=labels,
                                              batch_size=batch_size,
                                              image_size=image_size,
                                              augment=augment,
                                              random_seed=seed,
                                              label_enc=le,
                                              valid_size=valid_split,
                                              shuffle=shuffle,
                                              num_workers=4,
                                              pin_memory=False)

# # load ResNet50 with ImageNet weights
# model = resnet50(pretrained=True)

# # freeze all parameters
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(2048, num_classes)

# # send model to GPU
# model.cuda()

# # loss and optimizer
# crit = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
#                       lr=learning_rate, momentum=0.9, weight_decay=1e-6)

# # pre-train fc layer
# for epoch in range(num_epochs):
#     train(epoch, train_loader)
#     test(epoch, val_loader)

# torch.save(model.state_dict(), '../data/weights/resnet50_finetune-fc.pth')

# print('\nFinished.')

# load ResNet50 with ImageNet weights
model = resnet50(pretrained=False)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)

model.load_state_dict(torch.load('../data/weights/resnet50_finetune-fc-3-4.pth'))

# send model to GPU
model.cuda()

# # loss and optimizer
crit = nn.CrossEntropyLoss().cuda()

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
for epoch in range(num_epochs):
    train(epoch, train_loader)
    test(epoch, val_loader)
    torch.save(model.state_dict(), '../data/weights/resnet50_finetune-fc-3-4_lr-1e-4.pth')