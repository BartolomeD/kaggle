import os
import pickle
import itertools
import io
import bson
import threading
from scipy.ndimage import imread
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from resnet import *
import time
start_time = time.time()


# writes loss to file
def write_loss(x):
    with open('loss.txt', 'a') as f:
        f.write(x+'\n')


# helper to generate batch of data from source bson data
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# threadsafe 
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


# decorator to wrap generator with threadsafe           
def threadsafe(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


# batch generator to process and yield batch from source bson data
@threadsafe
def batch_generator(data_path, batch_size=32, return_labels=True):
    
    # decode data
    documents = bson.decode_file_iter(open(data_path, 'rb'))
    for batch in grouper(batch_size, documents): 
        x = []
        y = []
        for elem in batch:
            
            # get features
            img = elem.get('imgs')[0]
            img = imread(io.BytesIO(img.get('picture', None)))
            
            # get label
            category = elem.get('category_id', '')
            label = labelencoder.transform([category]) if category else None

            # process and append
            img = cv2.resize(img.astype('float32') / 255.0, (160, 160))
            x.append(img)
            y.append(label)

        x = Variable(torch.from_numpy(np.array(x).transpose(0, 3, 1, 2)).cuda())
        y = Variable(torch.from_numpy(np.array(y).ravel()).cuda())

        if return_labels:
            yield x, y
        else:
            yield x


# load labelencoder and categories if they exist
if os.path.isfile('../data/labelencoder.pkl'):
    with open('../data/labelencoder.pkl', 'rb') as f:
        labelencoder = pickle.load(f)
    categories = pd.read_csv('../data/categories.csv')

# else create label encoder and categories
else:
    documents = bson.decode_file_iter(open('../data/train.bson', 'rb'))
    categories = [(d['_id'], d['category_id']) for d in documents]
    categories = pd.DataFrame(categories, columns=['id', 'cat'])
    categories.to_csv('../data/categories.csv')
    labelencoder = LabelEncoder()
    labelencoder.fit(categories.cat.unique().ravel())
    with open('../data/labelencoder.pkl', 'wb') as f:
        pickle.dump(labelencoder, f)


# parameters
batch_size = 32
learning_rate = 1e-5
epochs = 3
num_classes = len(labelencoder.classes_)
val_split = round(0.9*(7069896//batch_size))

# model; freeze conv layer and train fully connected
model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, num_classes)
model = model.cuda()

# loss and optimizer
crit = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()

        write_loss(format(loss.data[0], '.4f'))

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] - Loss: {:.6f}'.format(
                epoch, batch_idx * batch_size, 7069896,
                100. * batch_idx / (7069896//batch_size), loss.data[0]))  
        if batch_idx >= val_split:
            return

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):

        output = model(data)
        test_loss += crit(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= (batch_idx + 1) * batch_size
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, (correct / ((batch_idx + 1) * batch_size))*100)) # 100. * correct / (batch_idx + 1) * batch_size))

for e in range(epochs):
    data_loader = batch_generator('../data/train.bson', batch_size=batch_size)
    train(e)
    test()
    torch.save(model.state_dict(), './resnet50_{}ep_finetuneClf.pth'.format(e+1))

print('\nFinished in {}s.'.format(format(time.time() - start_time, '.1f')))

# # train the model
# for e in range(epochs):
#     preds=[]
#     trues=[]
#     correct = 0
#     train_gen = batch_generator('../data/train.bson', batch_size=batch_size)
#     for i, (x, y) in enumerate(train_gen, 1):

#         # train
#         if i < 10:
#             optimizer.zero_grad()
#             out = model(x)
#             loss = crit(out, y.long())
#             loss.backward()
#             optimizer.step()

#         # validate
#         else:
#             _, pred = torch.max(model(x), -1)
#             correct += (pred == y).double().sum().data[0]

#             # preds.append([np.argmax(xi) for xi in model(x).data.cpu().numpy()])
#             # trues.append(y.data.cpu().numpy())
#             if i > 20:
#                 break

#         # save loss
#         loss_ = format(loss.data[0], '.4f')
#         # write_loss(loss_)
        
#         # print batch and epoch statistics
#         if i % 10 == 0:
#             print('\rEpoch {}/{} - Batch {}/{} - loss: {}'.format(
#                 e+1, epochs, i, 7069896//batch_size, loss_), end='')

#         prec1 = accuracy(out.data, y.data)
#         print(prec1)
#     break
#     np.savetxt('ep{}_preds.out'.format(e), np.array(preds).ravel(), delimiter=',')
#     np.savetxt('ep{}_trues.out'.format(e), np.array(trues).ravel(), delimiter=',')



# # save the model
# torch.save(model.state_dict(), './resnet50_3ep_finetuneClf.pth')

