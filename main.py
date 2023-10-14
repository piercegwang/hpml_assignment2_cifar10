'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from time import perf_counter

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--cuda', default=True, type=bool, help='Enable or disable cuda')
parser.add_argument('--data-path', default='./data', help='path to load CIFAR10 data from',
                    dest='data_path')
parser.add_argument('--dlw', default=2, type=int, help='Number of dataloader workers')
parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer to use')
args = parser.parse_args()

if not torch.cuda.is_available() and args.cuda:
    exit()

device = 'cuda' if args.cuda else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

dataloader_time = [0,0]
minibatch_training_time = [0,0]
total_running_time = [0,0]

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Training
def train(epoch, trainloader, net, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    minibatch_training_time = [0, 0]
    minibatch_training_time[0] = perf_counter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    minibatch_training_time[1] = perf_counter()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg Minibatch Training: %.3f s'
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total,
             (minibatch_training_time[1] - minibatch_training_time[0])/len(trainloader)))


def test(epoch, testloader, net, optimizer, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=False, download=True, transform=transform_test)

def main():
    global best_acc, start_epoch
    global dataloader_time, minibatch_training_time, total_running_time
    global trainset, testset
    print('-- Creating dataloaders with num_workers = %d' % args.dlw)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd+n":
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr,
                                weight_decay=5e-4)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr,
                                weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                                weight_decay=5e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+5):
        dataloader_time[0] = perf_counter()
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=args.dlw)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=args.dlw)
        dataloader_time[1] = perf_counter()

        total_running_time[0] = perf_counter()
        train(epoch, trainloader, net, optimizer, criterion)
        test(epoch, testloader, net, optimizer, criterion)
        scheduler.step()
        total_running_time[1] = perf_counter()
        print('-> Data Loading Time: %.3f s' % (dataloader_time[1] - dataloader_time[0]))
        print('-> Total Running Time: %.3f s' % (total_running_time[1] - total_running_time[0]))

main()
