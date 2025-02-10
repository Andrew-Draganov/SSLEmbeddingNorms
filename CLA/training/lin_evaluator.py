# Modified from https://github.com/p3i0t/SimCLR-CIFAR10/tree/master

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from CLA.utils.data_utils import get_training_loaders, NUM_CLASSES_DICT
from CLA.utils.utils import AverageMeter
from CLA.utils.knn_monitor import knn_monitor
from CLA.training.models import SimCLR

class Classifier(nn.Module):
    """Linear wrapper of encoder."""
    def __init__(self, encoder, feature_dim, n_classes):
        super().__init__()
        self.enc = encoder
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.enc(x))


def run_epoch(model, dataloader, epoch, optimizer=None):
    # Run a single epoch of the linear finetuning
    if optimizer:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('acc')
    loader_bar = tqdm(dataloader)
    for x, y in loader_bar:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))
        if optimizer:
            loader_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}" .format(
                epoch,
                loss_meter.avg,
                acc_meter.avg
            ))
        else:
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}" .format(
                epoch,
                loss_meter.avg,
                acc_meter.avg
            ))

    return loss_meter.avg, acc_meter.avg


def get_accuracies(args, pre_model, pre_model_str, load_epoch, knn=True):
    # Prepare data loaders and model
    _, finetune_loader, test_loader = get_training_loaders(args, args.dataset)
    pre_model.load_state_dict(torch.load('{}_epoch{}.pt'.format(pre_model_str, load_epoch)))

    # Train knn classifier on model
    knn_acc = knn_monitor(
        pre_model.backbone.cuda(),
        args.dataset,
        finetune_loader,
        test_loader,
        k=min(200, len(finetune_loader.dataset)),
        hide_progress=False
    )

    # Get linear classifier model from self-supervised backbone
    model = Classifier(
        pre_model.backbone,
        feature_dim=pre_model.backbone.output_dim,
        n_classes=NUM_CLASSES_DICT[args.dataset]
    )
    model = model.cuda()
    model.enc.requires_grad = False # Freeze the backbone
    parameters = list(model.classifier.parameters())  # trainable parameters.
    optimizer = torch.optim.SGD(parameters, args.lin_learning_rate, momentum=args.momentum, nesterov=args.nesterov)

    # Train linear classifier while keeping backbone frozen
    optimal_loss, optimal_acc = 1e5, 0.
    train_losses, train_accs, test_losses, test_accs = [np.zeros(args.finetune_epochs) for _ in range(4)]
    for epoch in range(1, args.finetune_epochs + 1):
        train_losses[epoch-1], train_accs[epoch-1] = run_epoch(model, finetune_loader, epoch, optimizer)
        test_losses[epoch-1], test_accs[epoch-1] = run_epoch(model, test_loader, epoch)

        if train_losses[epoch-1] < optimal_loss:
            optimal_loss = train_losses[epoch-1]
            optimal_acc = test_accs[epoch-1]
            torch.save(model.state_dict(), '{}_lin_best.pth'.format(pre_model_str))

    lin_finetune_results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }
    return knn_acc, lin_finetune_results


if __name__ == '__main__':
    finetune()
