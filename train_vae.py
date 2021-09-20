from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys
import torch.autograd as autograd
import torchvision.models as models
sys.path.append('.')

from vae import *
from set import *
from load_imagenet import imagenet, load_data, ImageNet100


def reconst_images(batch_size=64, batch_num=1, dataloader=None, model=None):
    cifar10_dataloader = dataloader
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                _, gx, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(gx[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_GX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - gx[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                wandb.log({"_Batch_{batch}_RX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
    print('reconstruction complete!')


def test(epoch, model, testloader):
    # set model as testing mode
    model.eval()
    acc_gx_avg = AverageMeter()
    acc_rx_avg = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(bs, -1)), p=2, dim=1)
            _, gx, _, _ = model(x)
            acc_gx = 1 - F.mse_loss(torch.div(gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / bs
            acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / bs

            acc_gx_avg.update(acc_gx.data.item(), bs)
            acc_rx_avg.update(acc_rx.data.item(), bs)

        wandb.log({'acc_gx_avg': acc_gx_avg.avg, \
                   'acc_rx_avg': acc_rx_avg.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tRec_gx: %.4f Rec_rx: %.4f " % (epoch, acc_gx_avg.avg, acc_rx_avg.avg))
        reconst_images(batch_size=64, batch_num=2, dataloader=testloader, model=model)
        torch.save(model.module.state_dict(),
                   os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        print("Epoch {} model saved!".format(epoch + 1))


def main(args):
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    print('\n[Phase 1] : Data Preparation')

    if args.dataset == 'tinyimagenet' or args.dataset == 'imagenet100' :
        size = 224
        normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        model = CVAE_imagenet_withbn(128, args.dim)
    else:
        size = 32
        normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        model = CVAE_cifar_withbn(128, args.dim)

    if args.simclr:
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        transform_train = transforms.Compose(
            [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalizer
            ]
        )
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])
    if args.dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'tinyimagenet':
        print("| Preparing Tiny-Imagenet dataset...")
        sys.stdout.write("| ")
        trainset, _ = load_data('../data/tiny_imagenet.pickle')
        trainset = imagenet(trainset, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'imagenet100':
        print("| Preparing imagenet100 dataset...")
        sys.stdout.write("| ")
        dataset = ImageNet100(data_path='/gpub/imagenet_raw', transform_train=transform_train)
        trainloader, _ = dataset.make_loaders(workers=4, batch_size=args.batch_size)

    # Model
    print('\n[Phase 2] : Model setup')
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()},
    ], lr=args.lr, betas=(0.0, 0.9))

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - epoch / args.epochs)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))
    print('| Initial Learning Rate = ' + str(args.lr))

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()

        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_kl = AverageMeter()

        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.cuda(), y.cuda().view(-1, )
            x, y = Variable(x), Variable(y)
            bs = x.size(0)

            _, gx, mu, logvar = model(x)

            optimizer.zero_grad()
            l_rec = F.mse_loss(x, gx)
            l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l_kl /= bs * 3 * args.dim
            loss = l_rec + args.kl * l_kl
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.data.item(), bs)
            loss_rec.update(l_rec.data.item(), bs)
            loss_kl.update(l_kl.data.item(), bs)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx
            wandb.log({'loss': loss_avg.avg, \
                       'loss_rec': loss_rec.avg, \
                       'loss_kl': loss_kl.avg, \
                       'lr': optimizer.param_groups[0]['lr']}, step=n_iter)
            if (batch_idx + 1) % 30 == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t Loss_rec: %.4f Loss_kl: %.4f'
                    % (epoch, args.epochs, batch_idx + 1,
                       len(trainloader),  loss_rec.avg, loss_kl.avg))
        scheduler.step()
        if epoch % 10 == 1:
            test(epoch, model, trainloader)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--dim', default=128, type=int, help='CNN_embed_dim')
    parser.add_argument('--kl', default=0.1, type=float, help='kl weight')
    parser.add_argument('--simclr', default=False, type=str, help='simclr')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("./results/", ''))
    set_random_seed(args.seed)
    main(args)
