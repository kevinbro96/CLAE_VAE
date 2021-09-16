import os
import torch
import torchvision
import argparse
import sys
from torch.autograd import Variable
import numpy as np
import wandb

from model import load_model, save_model
from modules import NT_Xent
from modules.transformations import TransformsSimCLR
from modules.transformations import TransformsSimCLR_imagenet
from utils import mask_correlated_samples
from load_imagenet import imagenet, load_data
sys.path.append('.')
sys.path.append('..')
from vae import *
from set import *

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=100, type=int,help='epochs')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
parser.add_argument('--projection_dim', default=64, type=int,help='projection_dim')
parser.add_argument('--optimizer', default="Adam", type=str, help="optimizer")
parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='weight_decay')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature')
parser.add_argument('--model_path', default='log/', type=str, 
                    help='model save path')
parser.add_argument('--model_dir', default='checkpoint/', type=str, 
                    help='model save path')

parser.add_argument('--dataset', default='CIFAR10',  
                    help='[CIFAR10, CIFAR100, tinyImagenet]')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='batch norm momentum for advprop')
parser.add_argument('--alpha', default=1.0, type=float, help='weight for contrastive loss with adversarial example')
parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
parser.add_argument('--vae_path',
                    default='../results/autoaug_new_8_0.5/model_epoch132.pth',
                    type=str, help='vae_path')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dim', default=128, type=int, help='CNN_embed_dim')
args = parser.parse_args() 
set_random_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def gen_adv(model, vae, x_i, criterion):
    x_i = x_i.detach()
    h_i, z_i = model(x_i, adv=True)

    with torch.no_grad():
        z, gx, _, _ = vae(x_i)
    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, True)
    x_j_adv = adv_gx + (x_i - gx).detach()
    h_j_adv, z_j_adv = model(x_j_adv, adv=True)
    tmp_loss = criterion(z_i, z_j_adv)
    tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle = variable_bottle + args.eps * sign_grad
        adv_gx = vae(variable_bottle, True)
    x_j_adv = adv_gx + (x_i - gx).detach()

    return x_j_adv.detach(), gx.detach()


def train(args, train_loader, model, vae, criterion, optimizer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        if args.adv:
            x_j_adv, gx = gen_adv(model, vae, x_i, criterion)
    
        optimizer.zero_grad()
        h_j, z_j = model(x_j)
        loss_og = criterion(z_i, z_j)
        loss = loss_og
        if args.adv:
            _, z_j_adv = model(x_j_adv, adv=True)
            loss_adv = criterion(z_i, z_j_adv)
            loss += args.alpha * loss_adv
        else:
            loss_adv = loss_og
        
        loss.backward()

        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        loss_epoch += loss.item()
        args.global_step += 1
        
        if args.debug:
            break
        if step % 10 == 0:
            wandb.log({'loss_og': loss_og.item(),
                       'loss_adv': loss_adv.item(),
                       'lr': optimizer.param_groups[0]['lr']})

        if args.global_step % 3000 == 0:
            if args.adv:
                reconst_images(x_i, gx, x_j_adv)

    return loss_epoch


def main():
    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_sampler = None
    if args.dataset == "CIFAR10":
        root = "../../data"
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimCLR()
        )
        data = 'non_imagenet'
  
    elif args.dataset == "CIFAR100":
        root = "../../data"
        train_dataset = torchvision.datasets.CIFAR100(
            root, download=True, transform=TransformsSimCLR()
        )
        data = 'non_imagenet'
    elif args.dataset == "tinyImagenet":
        root = '../datasets/tiny_imagenet.pickle'
        train_dataset, _ = load_data(root)
        train_dataset = imagenet(train_dataset, transform=TransformsSimCLR_imagenet(size=224))  
        data = 'imagenet'
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    log_dir = "log/" + args.dataset + '_log/'
    
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    if args.adv:
        suffix = suffix + '_alpha_{}_adv_eps_{}'.format(args.alpha, args.eps)
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag = True, bn_adv_momentum = args.bn_adv_momentum, data=data)
    else:
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag = False, bn_adv_momentum = args.bn_adv_momentum, data=data)

    vae = CVAE_cifar_withbn(128, args.dim)
    vae.load_state_dict(torch.load(args.vae_path))
    vae.to(device)
    vae.eval()

    suffix = suffix + '_proj_dim_{}'.format(args.projection_dim)
    suffix = suffix + '_bn_adv_momentum_{}_seed_{}'.format(args.bn_adv_momentum, args.seed)
    wandb.init(config=args, name=suffix.replace("_log/", ''))

    test_log_file = open(log_dir + suffix + '.txt', "w") 
    
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_dir = args.model_dir + args.dataset + '/'
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
            
    mask = mask_correlated_samples(args)
    criterion = NT_Xent(args.batch_size, args.temperature, mask, args.device)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(0, args.epochs):
        loss_epoch = train(args, train_loader, model, vae, criterion, optimizer)

        if scheduler:
            scheduler.step()
        print('epoch: {}% \t (loss: {}%)'.format(epoch,  loss_epoch/ len(train_loader)), file = test_log_file)
        test_log_file.flush()
        
        args.current_epoch += 1
        if args.debug:
            break    

    save_model(args.model_dir + suffix, model, optimizer, args.epochs)


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


if __name__ == "__main__":
    main()
