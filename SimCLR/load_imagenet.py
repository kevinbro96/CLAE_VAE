import os
import pickle
from PIL import Image
from robustness.datasets import CIFAR, DATASETS, DataSet
from robustness import data_augmentation as da
from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
import torch


class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, transform, train):
        super(MiniImageNet, self).__init__()
        if train:
            self.name = 'train'
        else:
            self.name = 'test'
        with open(os.path.join(root, '{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.imgs = data_dict['images']
        self.labels = data_dict['labels']
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, label = self.imgs[i], self.labels[i]
        if not torch.is_tensor(img):
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, label


def imagenet100(data_path, transform):
    custom_grouping = [[label] for label in range(0, 1000, 10)]
    ds_name = 'custom_imagenet'
    label_mapping = get_label_mapping(ds_name, custom_grouping)

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'val')

    train_set = folder.ImageFolder(root=train_path, transform=transform,
                                   label_mapping=label_mapping)
    test_set = folder.ImageFolder(root=test_path, transform=transform,
                                  label_mapping=label_mapping)
    return train_set, test_set


class imagenet:
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
        return (img, label)


def load_data(pickle_filename):
    with open(pickle_filename, "rb") as input_file:
        data = pickle.load(input_file)        
    return data['train'], data['val']
