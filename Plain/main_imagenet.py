'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import numpy as np
import models
import datasets
import math
import wandb

from utils import *
from load_imagenet import imagenet, load_data, MiniImageNet

sys.path.append('.')
sys.path.append('..')
from vae import *
from set import *
import models
from apex import amp

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
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

parser.add_argument('--resnet', default='resnet18',  help='resnet18, resnet34, resnet50, resnet101')
parser.add_argument('--dataset', default='tinyImagenet',  help='[tinyImagenet]')
parser.add_argument('--method', default='normal', type=str, help='adversarial exmaple')
parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--alpha', default=1.0, type=float, help='stregnth for regularization')
parser.add_argument('--debug', default=False, action='store_true', help='test_both_adv')
parser.add_argument('--vae_path',
                    default='../results/autoaug_new_8_0.5/model_epoch132.pth',
                    type=str, help='vae_path')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dim', default=128, type=int, help='CNN_embed_dim')
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt_level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
args = parser.parse_args()
set_random_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def gen_adv(net, vae, x):
    with torch.no_grad():
        z, gx, _, _ = vae(x)

    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, True)
    inputs_adv = adv_gx + (x - gx).detach()
    # generate adversarial example
    tmp_loss = CL(net, x, inputs_adv, gen_adv=True)
    if args.amp:
        with amp.scale_loss(tmp_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle = variable_bottle + args.eps * sign_grad
        adv_gx = vae(variable_bottle, True)
        inputs_adv = adv_gx + (x - gx).detach()
    inputs_adv.detach()
    return inputs_adv, gx


def gen_adv_clae(net, x):
    x.detach()
    inputs1_adv = Variable(x, requires_grad=True).to(device)
    # generate adversarial example
    tmp_loss = CL(net, x, inputs1_adv, gen_adv = True)
    if args.amp:
        with amp.scale_loss(tmp_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        tmp_loss.backward()
    inputs1_adv.data = inputs1_adv.data + (args.eps * torch.sign(inputs1_adv.grad.data))
    inputs1_adv.grad.data.zero_()
    inputs1_adv.detach()
    return inputs1_adv


def CL(model, x1, x2, adv=False, gen_adv = False):
    if gen_adv:
        x1_feat = model(x1, adv=True)
        x2_feat = model(x2, adv=True)
    else:
        x1_feat = model(x1)
        x2_feat = model(x2, adv=adv)
    #prototype set 1
    x1_x2_mat = torch.exp(torch.matmul(x1_feat,x2_feat.t())/args.batch_t)
    denominator1 = torch.sum(x1_x2_mat, dim = 1).view(x1.shape[0],1)
    prob1 = x1_x2_mat/denominator1
    prob1_diag = torch.diag(prob1)
    loss = -torch.mean(torch.log(prob1_diag))
    
    return loss


dataset = args.dataset
    
log_dir = args.log_dir + dataset + '_log/'
test_epoch = args.test_epoch
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.isdir(args.model_dir + '/' + dataset):
    os.makedirs(args.model_dir + '/' + dataset)
     
suffix = args.dataset + '_{}_batch_{}_embed_'.format(args.resnet, args.batch_size)
suffix = suffix + 'dim{}'.format(args.dim)

suffix = suffix + '_method_{}_eps_{}_alpha_{}'.format(args.method, args.eps, args.alpha)
suffix = suffix + '_bn_adv_momentum_{}_seed_{}'.format(args.bn_adv_momentum, args.seed)

wandb.init(config=args, name='train'+suffix.replace("_log/", ''))
  
if len(args.resume)>0:
    suffix = suffix + '_r'

# log the output
test_log_file = open(log_dir + suffix + '.txt', "w")                
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')

if args.dataset == "tinyImagenet":
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    root = '../../data/tiny_imagenet.pickle'
    trainset, _ = load_data(root)
    trainset = imagenet(trainset, transform=transform_train)
    vae = CVAE_imagenet_withbn(128, args.dim)

elif args.dataset == 'miniImagenet':
    root = '../../data'
    data = 'imagenet'
    trainset = MiniImageNet(root=root, transform=TransformsNormal_imagenet(), train=True)
    transform_test = transforms.Compose([
        transforms.Resize(size=[84, 84]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    testset = MiniImageNet(root=root, transform=transform_test, train=False)
    vae = CVAE_miniImagenet_withbn(128, args.dim)

else:
    raise NotImplementedError

trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last =True)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100, shuffle=False, num_workers=4)

ndata = trainset.__len__()

print('==> Building model..')
if args.method != 'normal':
    net = models.__dict__['MyResNet'](args.resnet, low_dim=args.low_dim, bn_adv_flag=True, bn_adv_momentum = args.bn_adv_momentum)
else:
    net = models.__dict__['MyResNet'](args.resnet, low_dim=args.low_dim, bn_adv_flag=False)

# define leminiscate: inner product within each mini-batch (Ours)

if device == 'cuda':
    cudnn.benchmark = True

vae.load_state_dict(torch.load(args.vae_path))

net.to(device)
vae.to(device)
vae.eval()

# define optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
if args.amp:
    [net, vae], optimizer = amp.initialize(
        [net, vae], optimizer, opt_level=args.opt_level)


def reconst_images(x_i, gx, x_j_adv):
    grid_X = torchvision.utils.make_grid(x_i[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"X.jpg": [wandb.Image(grid_X)]}, commit=False)
    grid_GX = torchvision.utils.make_grid(gx[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"GX.jpg": [wandb.Image(grid_GX)]}, commit=False)
    grid_RX = torchvision.utils.make_grid((x_i[32:96] - gx[32:96]).data, nrow=8, padding=2, normalize=True)
    wandb.log({"RX.jpg": [wandb.Image(grid_RX)]}, commit=False)
    grid_AdvX = torchvision.utils.make_grid(x_j_adv[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"AdvX.jpg": [wandb.Image(grid_AdvX)]}, commit=False)
    grid_delta = torchvision.utils.make_grid((x_j_adv - x_i)[32:96].data, nrow=8, padding=2, normalize=True)
    wandb.log({"Delta.jpg": [wandb.Image(grid_delta)]}, commit=False)
    wandb.log({'l2_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).norm(dim=1)),
               'linf_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).abs().max(dim=1)[0])
               }, commit=False)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed at 120, 160 and 200"""
    lr = args.lr
    if epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    elif epoch >= 160 and epoch <200:
        lr = args.lr * 0.05
    elif epoch >= 200:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, ((inputs1, inputs2), _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        
        if args.method == 'clae':
            inputs_adv = gen_adv_clae(net, inputs1)
        elif args.method == 'idaa':
            inputs_adv, gx = gen_adv(net, vae, inputs1)

        optimizer.zero_grad()
        loss_og = CL(net, inputs1, inputs2)
        loss = loss_og
        if args.method != 'normal':
            loss_adv = CL(net, inputs1, inputs_adv, adv=True)
            loss += args.alpha * loss_adv
        else:
            loss_adv = loss_og
            
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), 2 * inputs1.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % 10 == 0:
            wandb.log({'loss_og': loss_og.item(),
                       'loss_adv': loss_adv.item(),
                       'lr': optimizer.param_groups[0]['lr']})
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
        if args.global_step % 3000 == 0:
            if args.method == 'idaa':
                reconst_images(inputs1, gx, inputs_adv)
        if args.debug:
            break
        args.global_step += 1


args.global_step = 0
best_acc = 0   
for epoch in range(start_epoch, start_epoch+100):
    
    # training 
    train(epoch)
    print('----------Evaluation---------')
    start = time.time()
    acc = kNN_imagenet(epoch, net, trainloader, testloader, 200, args.batch_t, ndata, low_dim=args.low_dim)
    print("Evaluation Time: '{}'s".format(time.time() - start))
    if acc >= best_acc:
        print('Saving..')
        state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'acc': acc,
                }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, args.model_dir + '/' + dataset + '/' + suffix + '_best.t')
        best_acc = acc
    print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc))
    print('[Epoch]: {}'.format(epoch), file=test_log_file)
    print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc), file=test_log_file)
    wandb.log({'acc': acc})
    test_log_file.flush()
