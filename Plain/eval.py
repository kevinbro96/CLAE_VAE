'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

from torch.autograd import Variable
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import wandb
import os
import argparse
import time
import numpy as np
import models
import datasets
import math
sys.path.append('.')
sys.path.append('..')
from set import *
from utils import *


parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--log_dir', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str, 
                    help='model save path')
parser.add_argument('--test_epoch', default=1, type=int,
                    metavar='E', help='test every N epochs')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--batch-m', default=1, type=float,
                    metavar='N', help='m for negative sum')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--gpu', default='0,1,2,3', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--logistic_epochs', default=1000, type=int, metavar='B', help='training batch size')
parser.add_argument('--logistic_batch_size', default=128, type=int, metavar='B', help='training batch size')

parser.add_argument('--dataset', default='cifar10',  help='cifar10, cifar100')
parser.add_argument('--trial', type=int, help='trial')
parser.add_argument('--resnet', default='resnet18',  help='resnet18, resnet34, resnet50, resnet101')
parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
parser.add_argument('--eps', default=0.03, type=float, help='eps for adversarial')
parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--alpha', default=1.0, type=float, help='stregnth for regularization')
parser.add_argument('--debug', default=False, action='store_true', help='test_both_adv')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dim', default=128, type=int, help='CNN_embed_dim')
args = parser.parse_args() 
set_random_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset


class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


if args.dataset =='cifar10' or args.dataset =='cifar100':
    img_size = 32
    pool_len = 4
    
    
log_dir = args.log_dir  + args.dataset + '_eval_log/'
test_epoch = args.test_epoch
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)

if not os.path.isdir(args.model_dir + '/' + dataset + '_eval'):
    os.makedirs(args.model_dir + '/' + dataset + '_eval')

suffix = args.dataset + '_{}_batch_{}_embed_'.format(args.resnet, args.batch_size)
suffix = suffix + 'dim{}'.format(args.dim)
if args.adv:
    suffix = suffix + '_adv_eps_{}_alpha_{}'.format(args.eps, args.alpha)
    suffix = suffix + '_bn_adv_momentum_{}_seed_{}'.format(args.bn_adv_momentum, args.seed)
else:
    suffix = suffix + '_seed_{}'.format(args.seed)
wandb.init(config=args, name='LR'+suffix.replace("_log/", ''))
print(suffix)
# log the output
test_log_file = open(log_dir + suffix + '.txt', "w")                

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')


transform = transforms.Compose([
    transforms.Resize(size=img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

root='../../data'
if args.dataset == "cifar10":
    train_dataset = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=transform) 
    test_dataset = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=transform
    )
elif args.dataset == "cifar100":
    train_dataset = torchvision.datasets.CIFAR100(
        root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root, train=False, download=True, transform=transform
    )
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.logistic_batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=4,
)
    
ndata = train_dataset.__len__()

print('==> Building model..')
if args.adv:
    net = models.__dict__[args.resnet + '_cifar'](pool_len = pool_len, low_dim=args.low_dim, bn_adv_flag=True, bn_adv_momentum=args.bn_adv_momentum)
else:
    net = models.__dict__[args.resnet + '_cifar'](pool_len = pool_len, low_dim=args.low_dim, bn_adv_flag=False)

# define leminiscate: inner product within each mini-batch (Ours)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# Load checkpoint.
model_path = args.model_dir + '/' + dataset + '/' + suffix + '_best.t'
print('==> Load pretrained model {}'.format(model_path))

assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['net'])
net.to(device)
net.eval()
    
if args.dataset == "cifar100":
    n_classes = 100 # stl-10
else:
    n_classes = 10
model = LogisticRegression(args.low_dim, n_classes)
model = model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()


def train(loader, net, model, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    # switch to train mode
    net.eval()
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for batch_idx, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            feat = net(x)
        output = model(feat)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        
        if args.debug:
            break

    return loss_epoch, accuracy_epoch
        
    
def test(loader, net, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    net.eval
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            feat = net(x)
            output = model(feat)
            loss = criterion(output, y)
            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            loss_epoch += loss.item()
        if args.debug:
            break
    return loss_epoch, accuracy_epoch
        
best_acc = 0
for epoch in range(args.logistic_epochs):
    loss_epoch, accuracy_epoch = train(train_loader, net, model, criterion, optimizer)
    print("Train Epoch [{}]\t Loss: {}\t Accuracy: {}".format(epoch, loss_epoch / len(train_loader), accuracy_epoch / len(train_loader)), file = test_log_file)
    print("Train Epoch [{}]\t Loss: {}\t Accuracy: {}".format(epoch, loss_epoch / len(train_loader), accuracy_epoch / len(train_loader)))
    wandb.log({'Train/Loss': loss_epoch / len(train_loader),
               'Train/ACC': accuracy_epoch / len(train_loader)})
    test_log_file.flush()
    # final testing
    test_loss_epoch, test_accuracy_epoch = test(test_loader, net, model, criterion, optimizer)
    test_current_acc = test_accuracy_epoch / len(test_loader)
    if test_current_acc > best_acc:
        best_acc = test_current_acc
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(net, args.model_dir + '/' + dataset + '_eval/' + suffix + '_eval_best.t')
    print("Test Epoch [{}]\t Loss: {}\t Accuracy: {}\t Best Accuracy: {}".format(epoch, test_loss_epoch / len(test_loader), test_current_acc, best_acc), file = test_log_file)
    print("Test Epoch [{}]\t Loss: {}\t Accuracy: {}\t Best Accuracy: {}".format(epoch, test_loss_epoch / len(test_loader), test_current_acc, best_acc))
    wandb.log({'Test/Loss': test_loss_epoch / len(test_loader),
               'Test/ACC': test_current_acc,
              'Test/BestACC': best_acc})
    test_log_file.flush()
    
    if args.debug:
        break

print("Final \t Best Accuracy: {}".format(epoch, best_acc), file = test_log_file)
test_log_file.flush()


