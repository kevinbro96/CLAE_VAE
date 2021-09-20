import torch.nn as nn
import torchvision
from .resnet_BN import *
from .resnet_BN_imagenet import *
import pdb


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer3(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimSiam_BN(nn.Module):

    def __init__(self, args, bn_adv_flag=False, bn_adv_momentum=0.01, data='non_imagenet'):
        super(SimSiam_BN, self).__init__()

        self.args = args
        self.bn_adv_flag = bn_adv_flag
        self.bn_adv_momentum = bn_adv_momentum    
        if data == 'imagenet':
            self.backbone = self.get_imagenet_resnet(args.resnet)
        else:
            self.backbone = self.get_resnet(args.resnet)

        self.n_features = self.backbone.fc.in_features  # get dimensions of fc layer
        self.backbone.fc = Identity()  # remove fully-connected layer after pooling layer

        self.projector = projection_MLP(self.n_features)
        self.predictor = prediction_MLP()

    def get_resnet(self, name):
        resnets = {
            "resnet18": resnet18(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet34": resnet34(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet50": resnet50(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet101": resnet101(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet152": resnet152(pool_len=4, bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

    def get_imagenet_resnet(self, name):
        resnets = {
            "resnet18": resnet18_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet34": resnet34_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet50": resnet50_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet101": resnet101_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
            "resnet152": resnet152_imagenet(bn_adv_flag=self.bn_adv_flag, bn_adv_momentum=self.bn_adv_momentum),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

    def forward(self, x, adv=False):
        h = self.backbone(x, adv=adv)
        h = self.projector(h)
        z = self.predictor(h)
        return h, z
