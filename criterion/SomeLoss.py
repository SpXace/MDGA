from criterion import LSR
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from nfnets import SGD_AGC



import torch
import torch.nn as nn


'''
    fcanet
'''
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=6, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_v1(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # import pdb; pdb.set_trace()
        loss = (- targets * log_probs).mean(0).sum()
        return loss

    def forward_v2(self, inputs, targets):
        probs = self.logsoftmax(inputs)
        targets = torch.zeros(probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = nn.KLDivLoss()(probs, targets)
        return loss

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        return self.forward_v1(inputs, targets)

    def test(self):
        inputs = torch.randn(2, 5)
        targets = torch.randint(0, 5, [2])
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        print((targets * torch.log(targets) - targets * log_probs).sum(-1).mean())
        print(nn.KLDivLoss(reduce='mean')(log_probs, targets))
        # import pdb; pdb.set_trace()


class AverageMeter:
    def __init__(self):
        self.reset()

    def update(self, val):
        self.val += val.mean(0)
        self.num += 1

    def reset(self):
        self.num = 0
        self.val = 0

    def avg(self):
        try:
            return self.val / self.num
        except Exception:
            return None





def get_loss_dir(net_work_name, args):
    if "vgg".__eq__(net_work_name):
        return LSR()
    elif args.net == "densenet":
        return F.nll_loss
        #return nn.CrossEntropyLoss()
    elif args.net == 'fcanet':
        return CrossEntropyLabelSmooth()
    else:
        return nn.CrossEntropyLoss()


def get_optimizer(net_work_name, params, args):
    if "vgg".__eq__(net_work_name):
        return optim.SGD(params, lr=0.04, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif "densenet".__eq__(net_work_name):
        if args.opt == 'sgd':
            optimizer = optim.SGD(params, lr=0.01,
                                  momentum=0.9, weight_decay=1e-4)
            #optimizer = optim.SGD(params, lr=0.04, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif args.opt == 'adam':
            optimizer = optim.Adam(params, weight_decay=1e-4)
        elif args.opt == 'rmsprop':
            optimizer = optim.RMSprop(params, weight_decay=1e-4)
        return optimizer
    elif "mnasnet".__eq__(net_work_name) or "efficientnetv2".__eq__(net_work_name):
        optimizer = optim.RMSprop(params, lr=0.0001, momentum=0.9, weight_decay=1e-5)
        return optimizer
    elif "condensenet".__eq__(net_work_name):
        optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
        return optimizer
    elif "nfnet".__eq__(net_work_name):
        optimizer = SGD_AGC(
            named_params=params,  # Pass named parameters
            lr=0.1 * args.b / 256,
            momentum=0.9,
            clipping=0.1,  # New clipping parameter
            weight_decay=0.00002,
            nesterov=True)
        return optimizer
    elif "efficientnetv2".__eq__(net_work_name):
        optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
        return optimizer
    # elif "ghostnet".__eq__(net_work_name):
    #     optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    #     return optimizer
    elif "one".__eq__(net_work_name) or "two".__eq__(net_work_name):
        optimizer = optim.Adagrad(params, lr=0.001, weight_decay = 0.05)
        return optimizer
    else:
        optimizer = optim.SGD(params, lr=0.04, momentum=0.9, weight_decay=1e-4, nesterov=True)
        #optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)
        return optimizer