import os
import torch
import torchvision
import argparse
import sys
from torch.autograd import Variable
import numpy as np
import wandb
import torchvision.transforms as transforms
from model import load_model, save_model
from modules import NT_Xent
from modules.transformations import TransformsSimSiam
from modules.transformations import TransformsSimSiam_imagenet
from utils import mask_correlated_samples
from load_imagenet import imagenet, load_data
from eval_knn import kNN
sys.path.append('..')
from set import *
from vae import *


parser = argparse.ArgumentParser(description=' Seen Testing Category Training')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--dim', default=512, type=int, help='CNN_embed_dim')
parser.add_argument('--workers', default=4, type=int, help='workers')
parser.add_argument('--epochs', default=300, type=int, help='epochs')
parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
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
parser.add_argument('--seed', default=1, type=int, help='seed')

args = parser.parse_args()
print(args)
set_random_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def gen_adv(model, x_i):
    x_i = x_i.detach()
    h_i, z_i = model(x_i, adv=True)

    x_j_adv = Variable(x_i, requires_grad=True).to(args.device)
    h_j_adv, z_j_adv = model(x_j_adv, adv=True)
    tmp_loss = - F.cosine_similarity(z_j_adv, h_i.detach(), dim=-1).mean()
    tmp_loss.backward()

    x_j_adv.data = x_j_adv.data + (args.eps * torch.sign(x_j_adv.grad.data))
    x_j_adv.grad.data.zero_()

    x_j_adv.detach()
    x_j_adv.requires_grad = False
    return x_j_adv


def main():
    args.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_sampler = None
    if args.dataset == "CIFAR10":
        root = "../../data"
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimSiam()
        )
        data = 'non_imagenet'
        transform_test = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    elif args.dataset == "CIFAR100":
        root = "../../data"
        train_dataset = torchvision.datasets.CIFAR100(
            root, download=True, transform=TransformsSimSiam()
        )
        data = 'non_imagenet'
        transform_test = transforms.Compose([
            transforms.Resize(size=32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test)
    elif args.dataset == "tinyImagenet":
        root = '../../data/tiny_imagenet.pickle'
        train_dataset, testset = load_data(root)
        train_dataset = imagenet(train_dataset, transform=TransformsSimSiam_imagenet())
        data = 'imagenet'
        transform_test = transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        testset = imagenet(testset, transform=transform_test)
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
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=100, shuffle=False, num_workers=4)
    ndata = train_dataset.__len__()
    log_dir = "log/" + args.dataset + '_log/'

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    suffix = args.dataset + '_{}_batch_{}'.format(args.resnet, args.batch_size)
    if args.adv:
        suffix = suffix + '_alpha_{}_adv_eps_{}'.format(args.alpha, args.eps)
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag=True,
                                                 bn_adv_momentum=args.bn_adv_momentum, data=data)
    else:
        model, optimizer, scheduler = load_model(args, train_loader, bn_adv_flag=False,
                                                 bn_adv_momentum=args.bn_adv_momentum, data=data)

    suffix = suffix + '_bn_adv_momentum_{}_seed_{}'.format(args.bn_adv_momentum, args.seed)
    wandb.init(config=args, name=suffix.replace("_log/", ''))

    test_log_file = open(log_dir + suffix + '.txt', "w")

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_dir = args.model_dir + args.dataset + '/'
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    args.global_step = 0
    args.current_epoch = 0
    best_acc = 0
    for epoch in range(0, args.epochs):
        model.train()
        loss_epoch = 0
        for step, ((x_i, x_j), _) in enumerate(train_loader):

            optimizer.zero_grad()
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            # positive pair, with encoding
            h_i, z_i = model(x_i)
            if args.adv:
                x_j_adv = gen_adv(model, x_i)

            optimizer.zero_grad()
            h_j, z_j = model(x_j)
            loss_og = - F.cosine_similarity(z_i, h_j.detach(), dim=-1).mean() - F.cosine_similarity(z_j, h_i.detach(), dim=-1).mean()
            if args.adv:
                h_j_adv, z_j_adv = model(x_j_adv, adv=True)
                loss_adv = - F.cosine_similarity(z_i, h_j_adv.detach(), dim=-1).mean() - F.cosine_similarity(z_j_adv, h_i.detach(), dim=-1).mean()
                loss = loss_og + args.alpha * loss_adv
            else:
                loss = loss_og
                loss_adv = loss_og

            loss.mean().backward()

            optimizer.step()
            scheduler.step()
            if step % 50 == 0:
                print(
                    f"[Epoch]: {epoch} [{step}/{len(train_loader)}]\t Loss: {loss.item():.3f} Loss_og: {loss_og.item():.3f} Loss_adv: {loss_adv.item():.3f}")

            loss_epoch += loss.item()
            args.global_step += 1

            if args.debug:
                break
            if step % 10 == 0:
                wandb.log({'loss_og': loss_og.item(),
                           'loss_adv': loss_adv.item(),
                           'lr': optimizer.param_groups[0]['lr']})

        model.eval()
        print('epoch: {}% \t (loss: {}%)'.format(epoch, loss_epoch / len(train_loader)), file=test_log_file)

        print('----------Evaluation---------')
        start = time.time()
        acc = kNN(epoch, model.backbone, train_loader, testloader, 200, args.temperature, ndata, low_dim=512)
        print("Evaluation Time: '{}'s".format(time.time() - start))

        if acc >= best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(state, args.model_dir + suffix + '_best.t')
            best_acc = acc
        print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc))
        print('[Epoch]: {}'.format(epoch), file=test_log_file)
        print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc), file=test_log_file)
        wandb.log({'acc': acc})
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