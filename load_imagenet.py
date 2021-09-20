import pickle
from PIL import Image
import torch
import pdb
from robustness.datasets import CIFAR, DATASETS, DataSet, CustomImageNet


class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):

        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )


class imagenetEval:
    def __init__(self, data, transform = None):
        self.data = data
        self.imgs = data['imgs']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        label = self.labels[idx] 
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class imagenet:
    def __init__(self, data, transform = None, train=True):
        self.data = data
        self.imgs = data['imgs']
        self.labels = data['labels']
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        label = self.labels[idx] 
        if self.transform is not None:
            img1 = self.transform(img)
        return img1, label


def load_data(pickle_filename):
    with open(pickle_filename, "rb") as input_file:
        data = pickle.load(input_file)        
    return data['train'], data['val']
