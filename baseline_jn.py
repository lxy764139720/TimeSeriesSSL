from __future__ import print_function
import sys
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from cnn_gap import *
from sklearn.mixture import GaussianMixture
import wandb
import dataloader_jn as dataloader

parser = argparse.ArgumentParser(description='PyTorch JN Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=42)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./JN_dataset', type=str, help='path to dataset')
parser.add_argument('--dataset', default='jn', type=str)
args = parser.parse_args()

cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg = vars(args)
print(cfg)
wandb.init(project="TimeSeriesSSL_jn", config=cfg)
wandb.run.name = "baseline-" + str(cur_time)
wandb.run.save()
wandb.config["algorithm"] = "baseline"

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def warmup(epoch, net, optimizer, dataloader, flag):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1

    ce_loss = []
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()

        ce_loss.append(loss.item())
    wandb.log({(flag + " loss"): np.mean(ce_loss)}, step=epoch)


def test(epoch, net1):
    net1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()
    wandb.log({"Accuracy": acc}, step=epoch)


def create_model():
    model = Model(num_classes=args.num_class)
    model = model.cuda()
    return model


stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

loader = dataloader.jn_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                  num_workers=5, root_dir=args.data_path, log=stats_log,
                                  noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer1 = optim.Adam(net1.parameters(), lr=args.lr, weight_decay=5e-4)
wandb.config["optimizer"] = str(optimizer1).split(' ')[0]

CEloss = nn.CrossEntropyLoss()

all_loss = [[], []]  # save the history of losses from two networks

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')

    warmup_trainloader = loader.run('warmup')
    print('Warmup Net1')
    warmup(epoch, net1, optimizer1, warmup_trainloader, "net1")

    test(epoch, net1)
wandb.finish()
