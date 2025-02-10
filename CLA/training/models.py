import os
import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

def get_resnet(resnet_version):
    assert resnet_version in [18, 50]
    if resnet_version == 18:
        return resnet18
    return resnet50

def get_backbone(dataset, resnet_version):
    backbone = get_resnet(resnet_version)(zero_init_residual=True)

    if 'cifar' in dataset:
        # Customize for CIFAR10/100. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()

    backbone.output_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    return backbone

class Projector(nn.Module):
    def __init__(self, in_dim=512, out_dim=2048, batch_norm=True, n_layers=3):
        super().__init__()

        self.bn = batch_norm
        self.relu = nn.ReLU(inplace=True)

        if n_layers == 2:
            self.layer1 = nn.Linear(in_dim, out_dim)
            self.layer2 = nn.Linear(out_dim, out_dim)

            if self.bn:
                self.layer1_bn = nn.BatchNorm1d(out_dim)
                self.layer2_bn = nn.BatchNorm1d(out_dim)
            else:
                self.layer1_bn = nn.Identity()
                self.layer2_bn = nn.Identity()

            # Define the layers in this way so that cut_init has access to just the last layer if cut_all is false in config
            self.layers = nn.Sequential(
                self.layer1,
                self.layer1_bn,
                self.relu,
                self.layer2,
                self.layer2_bn,
            )


        else:
            self.layer1 = nn.Linear(in_dim, out_dim)
            self.layer2 = nn.Linear(out_dim, out_dim)
            self.layer3 = nn.Linear(out_dim, out_dim)

            if self.bn:
                self.layer1_bn = nn.BatchNorm1d(out_dim)
                self.layer2_bn = nn.BatchNorm1d(out_dim)
                self.layer3_bn = nn.BatchNorm1d(out_dim)
            else:
                self.layer1_bn = nn.Identity()
                self.layer2_bn = nn.Identity()
                self.layer3_bn = nn.Identity()

            # Define the layers in this way so that cut_init has access to just the last layer if cut_all is false in config
            self.layers = nn.Sequential(
                self.layer1,
                self.layer1_bn,
                self.relu,
                self.layer2,
                self.layer2_bn,
                self.relu,
                self.layer3,
                self.layer3_bn,
            )

    def forward(self, x):
        return self.layers(x)

class Predictor(nn.Module):
    def __init__(self, in_dim=2048, funnel_dim=512, out_dim=2048, batch_norm=True): # bottleneck structure
        super().__init__()
        self.funnel_dim = max(funnel_dim, 2)
        self.bn = batch_norm
        self.layer1 = nn.Linear(in_dim, self.funnel_dim)
        self.relu = nn.ReLU(inplace=True)
        if self.bn:
            self.layer1_bn = nn.BatchNorm1d(self.funnel_dim)
        else:
            self.layer1_bn = nn.Identity()
        self.layer2 = nn.Linear(self.funnel_dim, out_dim)

        self.layers = nn.Sequential(
            self.layer1,
            self.layer1_bn,
            self.relu,
            self.layer2
        )

    def forward(self, x):
        x = self.layer1(x)
        if self.bn:
            x = self.layer1_bn(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class SimCLR(nn.Module):
    def __init__(self, args):
        assert type(args.dataset) == str

        super().__init__()
        self.bn = args.bn

        self.feature_dim = args.projection_dim
        self.backbone = get_backbone(args.dataset, args.resnet_version)
        self.projector = Projector(
            self.backbone.output_dim,
            self.feature_dim,
            batch_norm=self.bn,
            n_layers=2
        )
        self.enc = nn.Sequential(self.backbone, self.projector)

    def forward(self, x):
        features = self.enc(x)
        return features


class SimSiam(nn.Module):
    def __init__(self, args):
        assert type(args.dataset) == str

        super(SimSiam, self).__init__()
        self.bn = args.bn

        self.backbone = get_backbone(args.dataset, args.resnet_version)
        self.embedding_dim = self.backbone.output_dim
        self.feature_dim = args.projection_dim
        self.enc = nn.Sequential(
            self.backbone,
            Projector(
                in_dim=self.embedding_dim,
                out_dim=self.feature_dim,
                batch_norm=self.bn,
            )
        )
        self.predictor = Predictor(
            in_dim=self.feature_dim,
            funnel_dim=int(self.feature_dim / 4),
            out_dim=self.feature_dim,
            batch_norm=self.bn
        )

    def forward(self, x):
        features = self.enc(x)
        prediction = self.predictor(features)
        features = features.detach()
        return features, prediction


class BYOL(nn.Module):
    def __init__(self, args):
        assert type(args.dataset) == str

        super(BYOL, self).__init__()
        self.tau = args.tau
        assert 0 <= self.tau <= 1
        self.bn = args.bn

        self.backbone = get_backbone(args.dataset, args.resnet_version)
        self.embedding_dim = self.backbone.output_dim
        self.feature_dim = args.projection_dim

        self.enc = nn.Sequential(
            self.backbone,
            Projector(
                in_dim=self.embedding_dim,
                out_dim=self.feature_dim,
                batch_norm=self.bn
            )
        )
        self.target_encoder = copy.deepcopy(self.enc)

        self.predictor = Predictor(
            in_dim=self.feature_dim,
            funnel_dim=int(self.feature_dim / 4),
            out_dim=self.feature_dim,
            batch_norm=self.bn
        )

    def forward(self, x):
        features = self.enc(x)
        prediction = self.predictor(features)
        return prediction

    def get_target(self, x):
        return self.target_encoder(x)

    @torch.no_grad()
    def update_moving_average(self):
        for online, target in zip(self.enc.parameters(), self.target_encoder.parameters()):
            target.data = self.tau * target.data + (1 - self.tau) * online.data


def get_default_model(model_str, args):
    assert model_str in ['simclr', 'simsiam', 'byol']
    if model_str == 'simclr':
        model = SimCLR(args).cuda()
    elif model_str == 'simsiam':
        model = SimSiam(args).cuda()
    else:
        model = BYOL(args).cuda()

    return model

