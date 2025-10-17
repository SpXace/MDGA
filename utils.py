
import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import transforms as vgg_transforms
from conf import settings
from dataset.dataset import Train_Seed_Classification_2021, Test_Seed_Classification_2021

def get_network(args):

    if args.net == 'vgg16':
        from models.vgg import vgg16
        net = vgg16()

    elif args.net == 'vgg11':
        from models.vgg import vgg11
        net = vgg11()

    elif args.net == 'vgg13':
        from models.vgg import vgg13
        net = vgg13()

    elif args.net == 'vgg19':
        from models.vgg import vgg19
        net = vgg19()

    elif args.net == 'densenet':
        from models.densenet import DenseNet
        net = DenseNet(growthRate=12, depth=100, reduction=0.5 ,bottleneck=True, nClasses=6)
    elif args.net == 'mnasnet':
        from models.MnasNet import MnasNet
        net = MnasNet()
    elif args.net == 'resnet18':
        from models.resnet import ResNet18
        net = ResNet18()
    elif args.net == 'resnet34':
        from models.resnet import ResNet34
        net = ResNet34()
    elif args.net == 'resnet50':
        from models.resnet import ResNet50
        net = ResNet50()
    elif args.net == 'resnet101':
        from models.resnet import ResNet101
        net = ResNet101()
    elif args.net == 'resnet152':
        from models.resnet import ResNet152
        net = ResNet152()
    elif args.net == 'sknet50':
        from models.sknet import SKNet50
        net = SKNet50()
    elif args.net == 'sknet101':
        from models.sknet import SKNet101
        net = SKNet101()
    elif args.net == 'densenetOne121':
        from models.densenetOne import DenseNet121
        net = DenseNet121()
    elif args.net == 'densenetOne169':
        from models.densenetOne import DenseNet169
        net = DenseNet169()
    elif args.net == 'densenetOne201':
        from models.densenetOne import DenseNet201
        net = DenseNet201()
    elif args.net == 'densenetOne161':
        from models.densenetOne import DenseNet161
        net = DenseNet161()
    elif args.net == 'condensenet':
        from models.condensenet import CondenseNet
        net = CondenseNet(args)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import MobileNet2
        net = MobileNet2(num_classes=6)
    elif args.net == 'shufflenet':
        from models.shufflenet import ShuffleNet
        net = ShuffleNet(num_classes=6)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import ShuffleNetV2
        net = ShuffleNetV2(num_classes=6)
    elif args.net == 'RegNetX_200MF':
        from models.regnet import create_regnet
        net = create_regnet(model_name=args.net, num_classes=6)
    elif args.net == 'RegNetX_400MF':
        from models.regnet import create_regnet
        net = create_regnet(model_name=args.net, num_classes=6)
    elif args.net == 'RegNetY_200MF':
        from models.regnet import create_regnet
        net = create_regnet(model_name=args.net, num_classes=6)
    elif args.net == 'RegNetY_400MF':
        from models.regnet import create_regnet
        net = create_regnet(model_name=args.net, num_classes=6)
    elif args.net == 'efficientnetv2_s':
        from models.efficientnetv2 import efficientnetv2_s
        net = efficientnetv2_s(num_classes=6)
    elif args.net == 'efficientnetv2_m':
        from models.efficientnetv2 import efficientnetv2_m
        net = efficientnetv2_m(num_classes=6)
    elif args.net == 'efficientnetv2_l':
        from models.efficientnetv2 import efficientnetv2_l
        net = efficientnetv2_l(num_classes=6)
    elif args.net == 'nfnet':
        from nfnets import NFNet
        net = NFNet()
    elif args.net == 'RepVGG_A0':
        from models.repvgg import create_RepVGG_A0
        net = create_RepVGG_A0(deploy=False)
    elif args.net == 'RepVGG_A1':
        from models.repvgg import create_RepVGG_A1
        net = create_RepVGG_A1(deploy=True)
    elif args.net == 'ghostnet':
        from models.ghostnet import ghostnet
        net = ghostnet()
    elif args.net == 'fcanet':
        from models.fcanet import fcanet34
        net = fcanet34(num_classes=6)
    return net

def get_train_dataloader(path, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to Train_Seed_Classification_2021 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    train_dataset = Train_Seed_Classification_2021(
        path,
        transform=transforms,
        target_transform=target_transforms
    )
    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
        # drop_last=True
    )

    return train_dataloader

def get_test_dataloader(path, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    test_dataset = Test_Seed_Classification_2021(
        path,
        transform=transforms,
        target_transform=target_transforms
    )

    test_dataloader =  DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
        # drop_last=True
    )

    return test_dataloader

def get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur

    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
        if 'bias' in name:
            last_layer_bias = para

    return last_layer_weights, last_layer_bias

def visualize_network(writer, net):
    """visualize network architecture"""
    input_tensor = torch.Tensor(3, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    input_tensor = input_tensor.to(next(net.parameters()).device)
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

def visualize_lastlayer(writer, net, n_iter):
    """visualize last layer grads"""
    weights, bias = get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)

def visualize_train_loss(writer, loss, n_iter):
    """visualize training loss"""
    writer.add_scalar('Train/loss', loss, n_iter)

def visualize_param_hist(writer, net, epoch):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def visualize_test_loss(writer, loss, epoch):
    """visualize test loss"""
    writer.add_scalar('Test/loss', loss, epoch)

def visualize_test_acc(writer, acc, epoch):
    """visualize test acc"""
    writer.add_scalar('Test/Accuracy', acc, epoch)

def visualize_learning_rate(writer, lr, epoch):
    """visualize learning rate"""
    writer.add_scalar('Train/LearningRate', lr, epoch)

def init_weights(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def mixup_data(x, y, alpha=0.2):

    """Returns mixed up inputs pairs of targets and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    index = index.to(x.device)

    lam = max(lam, 1 - lam)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a = y
    y_b = y[index, :]

    return mixed_x, y_a, y_b, lam

def get_train_transforms(net_work_name):
    if "vgg".__eq__(net_work_name):
        transform = vgg_transforms.Compose([
            vgg_transforms.ToCVImage(),
            vgg_transforms.RandomResizedCrop(settings.IMAGE_SIZE),
            vgg_transforms.RandomHorizontalFlip(),
            vgg_transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            vgg_transforms.ToTensor(),
            vgg_transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        ])
        return transform
    elif "densenet".__eq__(net_work_name):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(settings.NORMMEAN, settings.NORMSTD)
            # vgg_transforms.ToCVImage(),
            # vgg_transforms.RandomResizedCrop(settings.IMAGE_SIZE),
            # vgg_transforms.RandomHorizontalFlip(),
            # vgg_transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            # vgg_transforms.ToTensor(),
            # vgg_transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        ])
        return transform
    else:
        transform = transforms.Compose([
            transforms.Resize([settings.IMAGE_SIZE, settings.IMAGE_SIZE]),
            # transforms.Resize([13, 13]),
            transforms.ToTensor(),
            transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        ])
        return transform

def get_test_transforms(net_work_name):
    if "vgg".__eq__(net_work_name):
        transform = vgg_transforms.Compose([
            vgg_transforms.ToCVImage(),
            vgg_transforms.CenterCrop(settings.IMAGE_SIZE),
            vgg_transforms.ToTensor(),
            vgg_transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        ])
        # transform = transforms.Compose([
        #     transforms.Resize([settings.IMAGE_SIZE, settings.IMAGE_SIZE]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        # ])
        return transform
    elif "densenet".__eq__(net_work_name):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.NORMMEAN, settings.NORMSTD)
            # vgg_transforms.ToCVImage(),
            # vgg_transforms.CenterCrop(settings.IMAGE_SIZE),
            # vgg_transforms.ToTensor(),
            # vgg_transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        ])
        return transform
    else:
        transform = transforms.Compose([
            transforms.Resize([settings.IMAGE_SIZE, settings.IMAGE_SIZE]),
            # transforms.Resize([13, 13]),
            transforms.ToTensor(),
            transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
        ])
        return transform

