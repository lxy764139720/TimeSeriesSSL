import argparse
import shutil
import time
import datetime
import os
import os.path
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from cnn_dropout import Model
from dataloader_crwu_pencil import crwu_dataset, ComposeTransform, Jitter, Scaling, ToTensor

parser = argparse.ArgumentParser(description='PyTorch CRWU Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cnn_dropout',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--lr2', '--learning-rate2', default=0.005, type=float,
                    metavar='H-P', help='initial learning rate of stage3')
parser.add_argument('--alpha', default=0.4, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta', default=0.1, type=float,
                    metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=600, type=int,
                    metavar='H-P', help='the value of lambda')
parser.add_argument('--stage1', default=50, type=int,
                    metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=297, type=int,
                    metavar='H-P', help='number of epochs utill stage2')
parser.add_argument('--num_epochs', default=300, type=int, metavar='H-P',
                    help='number of total epochs to run')
parser.add_argument('--seed', default=42)
parser.add_argument('--dataset', default='crwu', type=str)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--classnum', default=10, type=int,
                    metavar='H-P', help='number of train dataset classes')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpuid', dest='gpuid', default=0, type=int,
                    help='select gpu')
parser.add_argument('--dir', dest='dir', default='./pencil/', type=str, metavar='PATH',
                    help='model dir')
parser.add_argument('--noise_mode', dest='noise_mode', default='sym', type=str,
                    help='noise mode')
parser.add_argument('--data_path', default='./CRWU_dataset', type=str, help='path to dataset')
args = parser.parse_args()

cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg = vars(args)
print(cfg)
wandb.init(project="TimeSeriesSSL_crwu", config=cfg)
wandb.run.name = "pencil-" + cur_time
wandb.run.save()
wandb.config["algorithm"] = "PENCIL"

best_prec1 = 0

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def main():
    global args, best_prec1

    y_file = args.dir + "y.npy"

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)
    if not os.path.isdir(args.dir + 'record'):
        os.makedirs(args.dir + 'record')

    model = Model().cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)

    checkpoint_dir = args.dir + "checkpoint.pth.tar"
    modelbest_dir = args.dir + "model_best.pth.tar"

    # optionally resume from a checkpoint
    # if os.path.isfile(checkpoint_dir):
    #     print("=> loading checkpoint '{}'".format(checkpoint_dir))
    #     checkpoint = torch.load(checkpoint_dir)
    #     args.start_epoch = checkpoint['epoch']
    #     # args.start_epoch = 0
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(checkpoint_dir))

    cudnn.benchmark = True

    # Data loading code
    transform1 = ComposeTransform([
        Jitter(),
        Scaling(),
        ToTensor(),
    ])
    transform2 = ComposeTransform([
        ToTensor(),
    ])
    trainset = crwu_dataset(r=args.r, root_dir='./CRWU_dataset', mode='train', noise_mode=args.noise_mode,
                            transform=transform1, noise_file='%s/%.1f_%s.json' %
                                                             (args.data_path, args.r, args.noise_mode))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers, pin_memory=True)
    testset = crwu_dataset(r=args.r, root_dir='./CRWU_dataset', mode='test', noise_mode=args.noise_mode,
                           transform=transform2, noise_file='%s/%.1f_%s.json' %
                                                            (args.data_path, args.r, args.noise_mode))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.num_epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # load y_tilde
        if os.path.isfile(y_file):
            y = np.load(y_file)
        else:
            y = []

        train(trainloader, model, criterion, optimizer, epoch, y)

        # evaluate on validation set
        prec1 = validate(testloader, model, criterion, epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, filename=checkpoint_dir, modelbest=modelbest_dir)


def train(train_loader, model, criterion, optimizer, epoch, y):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # new y is y_tilde after updating
    new_y = np.zeros([len(train_loader.dataset), args.classnum])

    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        index = index.numpy()

        target1 = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target1).cuda()

        # compute output
        output = model(input_var)

        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        softmax = nn.Softmax(dim=1).cuda()
        if epoch < args.stage1:
            # lc is classification loss
            lc = criterion(output, target_var)
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 10.0)
            onehot = onehot.numpy()
            new_y[index, :] = onehot
        else:
            yy = y
            yy = yy[index, :]
            yy = torch.FloatTensor(yy)
            yy = yy.cuda()
            yy = torch.autograd.Variable(yy, requires_grad=True)
            # obtain label distributions (y_hat)
            last_y_var = softmax(yy)
            lc = torch.mean(softmax(output) * (logsoftmax(output) - torch.log(last_y_var)))
            # lo is compatibility loss
            lo = criterion(last_y_var, target_var)
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        if epoch < args.stage1:
            loss = lc
        elif epoch < args.stage2:
            loss = lc + args.alpha * lo + args.beta * le
        else:
            loss = lc

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target1, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.stage1 <= epoch < args.stage2:
            lambda1 = args.lambda1
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1 * yy.grad.data)

            new_y[index, :] = yy.data.cpu().numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    if epoch < args.stage2:
        # save y_tilde
        y = new_y
        y_file = args.dir + "y.npy"
        np.save(y_file, y)
        y_record = args.dir + "record/y_%03d.npy" % epoch
        np.save(y_record, y)
    wandb.log({"Loss": losses.val}, step=epoch)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    wandb.log({"Accuracy": top1.val}, step=epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename='', modelbest=''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if epoch < args.stage2:
        lr = args.lr
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
