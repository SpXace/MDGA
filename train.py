import argparse
import glob
import logging
import math
import os
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import transforms
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from criterion import LSR
from criterion import get_loss_dir
from criterion import get_optimizer
import xlwt
import time

import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(107)
logging.basicConfig(filename=r'logs.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0, warmup=None, warmup_iters=None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter=None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters <= self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            if self.iters == self.warmup_iters:
                self.iters = 0
                self.warmup = None
            return
        
        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2


def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key)) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            #print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer

if __name__ == '__main__':
    startTime = time.time()
    net_work_name = 'your_model_name'
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-e', type=int, default=100, help='training epoches')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')

    if net_work_name.__eq__("vgg"):
        parser.add_argument('-net', type=str, default='vgg11', help='net type')
        parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
        parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    elif net_work_name.__eq__("densenet"):
        #densenet densenetOne121 densenetOne169 densenetOne201 densenetOne161
        parser.add_argument('-net', type=str, default='densenet', help='net type')
        parser.add_argument('-opt', type=str, default='sgd',choices=('sgd', 'adam', 'rmsprop'))
    elif net_work_name.__eq__("mnasnet"):
        parser.add_argument('-net', type=str, default='mnasnet', help='net type')
        parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
        parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    elif net_work_name.__eq__("last_resnet34") or net_work_name.__eq__("senet"):
        #resnet18 resnet34 resnet50 resnet101 resnet152
        parser.add_argument('-net', type=str, default='resnet34', help='net type')
        parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
        parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    elif net_work_name.__eq__("sknet"):
        #sknet50 sknet101
        parser.add_argument('-net', type=str, default='sknet50', help='net type')
        parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
        parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    elif net_work_name.__eq__("condensenet"):
        parser.add_argument('-net', type=str, default='condensenet', help='net type')
        # parser.add_argument('data', default='myself',
        #                     help='path to dataset')
        parser.add_argument('--stages', type=str, default='4-6-8-10-8', metavar='STAGE DEPTH',
                            help='per layer depth')
        parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
                            help='bottleneck (default: 4)')
        parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
                            help='1x1 group convolution (default: 4)')
        parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
                            help='3x3 group convolution (default: 4)')
        parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
                            help='condense factor (default: 4)')
        parser.add_argument('--growth', type=str, default='8-16-32-64-128', metavar='GROWTH RATE',
                            help='per layer growth')
        parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
                            help='transition reduction (default: 0.5)')
        parser.add_argument('--dropout-rate', default=0, type=float,
                            help='drop out (default: 0)')
        parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
                            help='group lasso loss weight (default: 0)')
        parser.add_argument('--evaluate', action='store_true',
                            help='evaluate model on validation set (default: false)')
        parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                            help='path to saved checkpoint (default: none)')
        parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                            help='path to saved checkpoint (default: none)')
    elif net_work_name.__eq__("mobilenetv2"):
        parser.add_argument('-net', type=str, default='mobilenetv2', help='net type')
    elif net_work_name.__eq__("shufflenet"):
        parser.add_argument('-net', type=str, default='shufflenet', help='net type')
    elif net_work_name.__eq__("shufflenetv2"):
        parser.add_argument('-net', type=str, default='shufflenetv2', help='net type')
    elif net_work_name.__eq__("last_regnet_X200"):
        #RegNetX_400MF RegNetX_200MF RegNetY_200MF RegNetY_400MF
        parser.add_argument('-net', type=str, default='RegNetX_200MF', help='net type')
    elif net_work_name.__eq__("efficientnetv2"):
        #efficientnetv2_s efficientnetv2_m efficientnetv2_l
        parser.add_argument('-net', type=str, default='efficientnetv2_s', help='net type')
    elif net_work_name.__eq__("nfnet"):
        parser.add_argument('-net', type=str, default='nfnet', help='net type')
    elif net_work_name.__eq__("repvgg"):
        # RepVGG_A0 RepVGG_A1
        parser.add_argument('-net', type=str, default='RepVGG_A0', help='net type')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--custwd', dest='custwd', action='store_true',
                            help='Use custom weight decay. It improves the accuracy and makes quantization easier.')
    elif net_work_name.__eq__("ghostnet"):
        parser.add_argument('-net', type=str, default='ghostnet', help='net type')
    elif net_work_name.__eq__("fcanet"):
        parser.add_argument('-net', type=str, default='fcanet', help='net type')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR',
                            help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
    elif net_work_name.__eq__("one"):
        #one two
        parser.add_argument('-net', type=str, default='one', help='net type')
    args = parser.parse_args()
    if net_work_name == 'condensenet':
        args.stages = list(map(int, args.stages.split('-')))
        args.growth = list(map(int, args.growth.split('-')))
        if args.condense_factor is None:
            args.condense_factor = args.group_1x1
        args.num_classes = 6

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, "your_model_name", settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    
    train_transforms = get_train_transforms(net_work_name)
    test_transforms = get_test_transforms(net_work_name)

    train_dataloader = get_train_dataloader(
        settings.TRAIN_DATA_PATH,
        train_transforms,
        args.b,
        args.w
    )

    test_dataloader = get_test_dataloader(
        settings.TEST_DATA_PATH,
        test_transforms,
        args.b,
        args.w
    )

    net = get_network(args)
    params = net.parameters()

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]

    net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()

    optimizer = get_optimizer(net_work_name, params, args)

    loss_dir = get_loss_dir(net_work_name, args)
    if net_work_name.__eq__("fcanet"):
        len_epoch = int(math.ceil(len(train_dataloader.dataset) / args.b))
        T_max = 95 * len_epoch
        warmup_iters = 5 * len_epoch
        scheduler = CosineAnnealingLR(optimizer, T_max, warmup='linear', warmup_iters=warmup_iters)
    best_acc = 0.0
    for epoch in range(1, args.e + 1):

        net.train()

        for batch_index, (images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            if (args.net == 'densenet'):
                images, labels = Variable(images), Variable(labels)
            if net_work_name.__eq__("fcanet"):
                scheduler.step()
            optimizer.zero_grad()
            predicts = net(images)
            loss = loss_dir(predicts, labels)
            if net_work_name.__eq__("repvgg"):
                if args.custwd:
                    for module in net.modules():
                        if hasattr(module, 'get_custom_L2'):
                            loss += args.weight_decay * 0.5 * module.get_custom_L2()
            loss.backward()
            optimizer.step()

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\t'.format(
                loss.item(),
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_dataloader.dataset),
            ))
            logging.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\t'.format(
                loss.item(),
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_dataloader.dataset),
            ))
        if net_work_name.__eq__("efficientnetv2"):
            scheduler.step()
        net.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.cuda()
                labels = labels.cuda()
                predicts = net(images)
                a = predicts.softmax(dim = 1)
                _, preds = predicts.max(1)
                correct += preds.eq(labels).sum().float()
                loss = loss_dir(predicts, labels)
                total_loss += loss.item()
            test_loss = total_loss / len(test_dataloader)
            acc = correct / len(test_dataloader.dataset)
            print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
            print()
            logging.info('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
