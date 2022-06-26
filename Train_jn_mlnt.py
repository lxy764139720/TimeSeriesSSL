import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from cnn_dropout_mlnt import Model
import math
import os
import sys
import time
import argparse
import datetime
from torch.autograd import Variable
from collections import OrderedDict
from dataloader_jn_mlnt import jn_dataloader
import random
import wandb

parser = argparse.ArgumentParser(description='PyTorch JN Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--meta_lr', default=0.005, type=float, help='meta learning_rate')
parser.add_argument('--num_fast', default=10, type=int, help='number of random perturbations')
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='ratio of random perturbations')
parser.add_argument('--start_iter', default=1, type=int)
parser.add_argument('--mid_iter', default=2, type=int)
parser.add_argument('--start_epoch', default=10, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--alpha', default=1, type=int)
parser.add_argument('--eps', default=0.99, type=float, help='Running average of model weights')
parser.add_argument('--seed', default=42)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='MLNT')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--data_path', default='./JN_dataset', type=str, help='path to dataset')
parser.add_argument('--dataset', default='jn', type=str)
args = parser.parse_args()

cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cfg = vars(args)
print(cfg)
wandb.init(project="TimeSeriesSSL_jn", config=cfg)
wandb.run.name = "mlnt-" + cur_time
wandb.run.save()
wandb.config["algorithm"] = "MLNT"

random.seed(args.seed)
torch.cuda.set_device(args.gpuid)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# Training
def train(epoch):
    global init, best
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0

    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate = learning_rate / 10
    if epoch > args.num_epochs * 0.8:
        learning_rate = learning_rate / 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' % (args.id, epoch, learning_rate))

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation

        class_loss = criterion(outputs, targets)  # Loss
        class_loss.backward(retain_graph=True)

        if batch_idx > args.start_iter or epoch > args.start_epoch:
            if batch_idx > args.mid_iter or epoch > args.start_epoch:
                args.eps = 0.999
                alpha = args.alpha
            else:
                u = (batch_idx - args.start_iter) / (args.mid_iter - args.start_iter)
                alpha = args.alpha * math.exp(-5 * (1 - u) ** 2)

            if init:
                init = False
                for param, param_tch in zip(net.parameters(), tch_net.parameters()):
                    param_tch.data.copy_(param.data)
            else:
                for param, param_tch in zip(net.parameters(), tch_net.parameters()):
                    param_tch.data.mul(args.eps).add(param.data, alpha=(1 - args.eps))

            _, feats = pretrain_net(inputs, get_feat=True)
            tch_outputs = tch_net(inputs, get_feat=False)
            p_tch = F.softmax(tch_outputs, dim=1)
            p_tch = p_tch.detach()

            for i in range(args.num_fast):
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                for n in range(int(targets.size(0) * args.perturb_ratio)):
                    num_neighbor = 10
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1, feat.size(0))
                    feat.data = feat.data.expand(targets.size(0), feat.size(0))
                    dist = torch.sum((feat - feats) ** 2, dim=1)
                    _, neighbor = torch.topk(dist.data, num_neighbor + 1, largest=False)
                    targets_fast[idx] = targets[neighbor[random.randint(1, num_neighbor)]]

                fast_loss = criterion(outputs, targets_fast)

                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True,
                                            only_inputs=True)
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False

                fast_weights = OrderedDict(
                    (name, param - args.meta_lr * grad) for ((name, param), grad) in zip(net.named_parameters(), grads))

                fast_out = net.forward(inputs, fast_weights)

                logp_fast = F.log_softmax(fast_out, dim=1)

                if i == 0:
                    consistent_loss = consistent_criterion(logp_fast, p_tch)
                else:
                    consistent_loss = consistent_loss + consistent_criterion(logp_fast, p_tch)

            meta_loss = consistent_loss * alpha / args.num_fast

            meta_loss.backward()

        optimizer.step()  # Optimizer update

        train_loss += class_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, args.num_epochs, batch_idx + 1, (len(train_loader.dataset) // args.batch_size) + 1,
                            class_loss.item(), 100. * correct / total))
        sys.stdout.flush()

        acc = 100. * correct / total
        if acc > best:
            best = acc
            print('| Saving Best Model (tchnet)...')
        save_point = './checkpoint/%s.pth.tar' % args.id
        save_checkpoint({
            'state_dict': net.state_dict(),
            'best_acc': best,
        }, save_point)
        wandb.log({"Loss": class_loss.item()}, step=epoch)


def test():
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = test_net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100. * correct / total
    print('* Test results : Acc@1 = %.2f%%' % test_acc)
    record.write('\nTest Acc: %f\n' % test_acc)
    record.flush()
    wandb.log({"Accuracy": test_acc}, step=epoch)


record = open('./checkpoint/' + args.id + '.txt', 'w')
record.write('learning rate: %f\n' % args.lr)
record.write('batch size: %f\n' % args.batch_size)
record.write('start iter: %d\n' % args.start_iter)
record.write('mid iter: %d\n' % args.mid_iter)
record.flush()

loader = jn_dataloader(batch_size=args.batch_size, num_workers=5, dataset=args.dataset, noise_mode=args.noise_mode,
                       r=args.r, root_dir=args.data_path,
                       noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))
train_loader, test_loader = loader.run()

best = 0
init = True
# Model
print('\nModel setup')
print('| Building net')
net = Model()
tch_net = Model()
pretrain_net = Model()
test_net = Model()

if use_cuda:
    net.cuda()
    tch_net.cuda()
    pretrain_net.cuda()
    test_net.cuda()
    cudnn.benchmark = True
pretrain_net.eval()

for param in tch_net.parameters():
    param.requires_grad = False
for param in pretrain_net.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
consistent_criterion = nn.KLDivLoss(reduction='batchmean')
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))

for epoch in range(1, 1 + args.num_epochs):
    train(epoch)

    print('\nTesting model')
    best_model = torch.load('./checkpoint/%s.pth.tar' % args.id)
    test_net.load_state_dict(best_model['state_dict'])
    test()

record.close()
