# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from dataloader_crwu_cotea import crwu_dataset, ToTensor
from cnn_gap import Model
import argparse, sys
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import datetime
import json
import random
import shutil
import wandb

from loss_cotea import loss_coteaching, loss_coteaching_plus

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--r', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_mode', type=str, help='[pairflip, sym, asym]', default='sym')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='crwu', default='crwu')
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--data_path', default='./CRWU_dataset', type=str, help='path to dataset')
args = parser.parse_args()

cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg = vars(args)
print(cfg)
wandb.init(project="TimeSeriesSSL_crwu", config=cfg)
wandb.run.name = "co_teaching+-" + cur_time
wandb.run.save()
wandb.config["algorithm"] = "co-teaching+"
wandb.config["architecture"] = "cnn_gap"

# Seed
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr


if args.dataset == 'crwu':
    init_epoch = 10
    train_dataset = crwu_dataset(dataset=args.dataset,
                               r=args.r,
                               mode='train',
                               transform=ToTensor(),
                               noise_mode=args.noise_mode,
                               noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode)
                               )

    test_dataset = crwu_dataset(dataset=args.dataset,
                              r=args.r,
                              mode='test',
                              transform=ToTensor(),
                              noise_mode=args.noise_mode,
                              noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode)
                              )

if args.forget_rate is None:
    forget_rate = args.r
else:
    forget_rate = args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * (args.n_epoch + 1)
beta1_plan = [mom1] * (args.n_epoch + 1)
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)

    # define drop rate schedule


def gen_forget_rate(fr_type='type_1'):
    if fr_type == 'type_1':
        rate_schedule = np.ones(args.n_epoch + 1) * forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    # if fr_type=='type_2':
    #    rate_schedule = np.ones(args.n_epoch)*forget_rate
    #    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    #    rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)

    return rate_schedule


rate_schedule = gen_forget_rate(args.fr_type)

save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % args.model_type + args.noise_mode + '_' + str(args.r)

txtfile = save_dir + "/" + model_str + ".txt"
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    print('Training %s...' % model_str)

    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0

    loss_ls1 = []
    loss_ls2 = []
    for i, (data, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        labels, data = labels.cuda(), data.cuda()

        # Forward + Backward + Optimize
        logits1 = model1(data)
        prec1, = accuracy(logits1, labels, topk=(1,))
        train_total += 1
        train_correct += prec1

        logits2 = model2(data)
        prec2, = accuracy(logits2, labels, topk=(1,))
        train_total2 += 1
        train_correct2 += prec2
        if epoch < init_epoch:
            loss_1, loss_2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        else:
            if args.model_type == 'coteaching_plus':
                loss_1, loss_2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], ind,
                                                            noise_or_not, epoch * i)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' \
                % (epoch + 1, args.n_epoch, i + 1, len(train_dataset) // batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

        loss_ls1.append(loss_1.cpu().detach().item())
        loss_ls2.append(loss_2.cpu().detach().item())

    train_acc1 = float(train_correct) / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total2)
    wandb.log({"net1 loss": np.mean(loss_ls1),
               "net2 loss": np.mean(loss_ls2)},
              step=epoch)
    return train_acc1, train_acc2


# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=args.num_workers,
                             shuffle=False)
    # Define models
    print('building model...')
    if args.dataset == 'crwu':
        clf1 = Model()

    clf1.cuda()
    print(clf1.parameters)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    if args.dataset == 'crwu':
        clf2 = Model()

    clf2.cuda()
    print(clf2.parameters)
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 train_acc2 test_acc1 test_acc2\n')

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    # evaluate models with random weights
    test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + " " + str(
            test_acc2) + "\n")

    # training
    for epoch in range(0, args.n_epoch + 1):
        # train models
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        train_acc1, train_acc2 = train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2)
        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        wandb.log({"net1 Accuracy": test_acc1,
                   "net2 Accuracy": test_acc2,
                   "mean Accuracy": (test_acc1 + test_acc2) / 2,
                   "Accuracy": max(test_acc1, test_acc2)},
                  step=epoch)

        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(train_acc2) + ' ' + str(test_acc1) + " " + str(
                    test_acc2) + "\n")


if __name__ == '__main__':
    main()
