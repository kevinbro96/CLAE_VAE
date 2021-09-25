import os
import torch
from modules import SimSiam_BN
import numpy as np
import pdb


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def load_model(args, loader, reload_model=False, load_path = None, bn_adv_flag=False, bn_adv_momentum = 0.01, data='non_imagenet'):

    model = SimSiam_BN(args, bn_adv_flag=bn_adv_flag, bn_adv_momentum = bn_adv_momentum, data = data)

    if reload_model:
        if os.path.isfile(load_path):
            model_fp = os.path.join(load_path)
        else:
            print("No file to load")
            return
        model.load_state_dict(torch.load(model_fp, map_location=lambda storage, loc: storage))
        
    model = model.to(args.device)

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': 3e-2*args.batch_size/256
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': 3e-2*args.batch_size/256
    }]

    optimizer = torch.optim.SGD(parameters, lr=3e-2*args.batch_size/256, momentum=0.9, weight_decay=0.0005)  # TODO: LARS
    scheduler = LR_Scheduler(
        optimizer,
        10, 0,
        args.epochs, 3e-2*args.batch_size/256, 0,
        len(loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    return model, optimizer, scheduler


def save_model(model_dir, model, optimizer, epoch):
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))
    else:
        torch.save(model.state_dict(), model_dir + '_epoch_{}.pt'.format(epoch))


def save_model_best(model_dir, model, optimizer):
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), model_dir + '_best.t')
    else:
        torch.save(model.state_dict(), model_dir + '_best.t')