import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PIL import Image


class StanfordDogsDataset():
    '''Stanford Dogs Dataset'''
    
    def __init__(self, csv_file, root_dir, label_enc=None, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.labenc = label_enc
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        label = str(self.labels.iloc[idx, 1])  
        if self.transform:
            image = self.transform(image)            
        if self.labenc:
            label = self.labenc.transform([label])            
        return {'image': image, 'label': label}


def train_valid_loader(data_dir,
                       labels,
                       batch_size,
                       image_size,
                       augment,
                       random_seed,
                       label_enc,
                       valid_size=0.1,
                       shuffle=True,
                       num_workers=4,
                       pin_memory=False):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over the CIFAR-10 dataset. A sample 
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Scale(size=image_size),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            normalize
        ])
    if augment:
        train_transform = transforms.Compose([
                transforms.RandomSizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

    else:
        train_transform = transforms.Compose([
            transforms.Scale(size=image_size),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    train_dataset = StanfordDogsDataset(csv_file=labels,
                                        root_dir=data_dir,
                                        label_enc=label_enc,
                                        transform=train_transform)

    valid_dataset = StanfordDogsDataset(csv_file=labels,
                                        root_dir=data_dir,
                                        label_enc=label_enc,
                                        transform=valid_transform)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, sampler=train_sampler, 
                              num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = DataLoader(valid_dataset, 
                              batch_size=batch_size, sampler=valid_sampler, 
                              num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader)
    
def test_loader(data_dir,
                labels,
                batch_size,
                image_size,
                shuffle=True,
                num_workers=4,
                pin_memory=False):
    """
    Utility function for loading and returning a multi-process 
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transform
    transform = transforms.Compose([
        transforms.Scale(size=image_size),
        transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        normalize
    ])

    dataset = StanfordDogsDataset(csv_file=labels,
                                  root_dir=data_dir,
                                  transform=transform)

    test_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    return test_loader
