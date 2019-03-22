from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import conv3x3, conv1x1, norm, ResBlock, ODEfunc, ODEBlock, PrimaryCaps, ConvCaps, CapsNet, capsules
from loss import SpreadLoss
from datasets import smallNORB

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                    help='dataset for training(mnist, smallNORB)')
parser.add_argument('--tol', type=float, default=.00001, metavar='TOL')


def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'mnist':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'smallNORB':
        num_class = 5
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    elif args.dataset == 'cifar10':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.RandomCrop(32),
                                 transforms.ToTensor()
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=False,
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.ToTensor()
                             ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    trainingdata = open("trainingdata.txt", "a")
    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)


        output = model(data)

        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc[0].item()
        if batch_idx % args.log_interval == 0:
            res = ('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.item(), acc[0].item(),
                  batch_time=batch_time, data_time=data_time))
            print(res)
            trainingdata.write(res)
    return epoch_acc


def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test = open("testdata.txt", "a")
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    test.write('Test Set: Average Loss: {:.6f}, Accuracy: {:.6f} \n'.format(test_loss, acc))
    return acc


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model\
    """
    downsampling_layers = [
            nn.Conv2d(1, 32, 3, 1),
            ResBlock(32, 32, stride=2, downsample=conv1x1(32, 32, 2)),
            ResBlock(32, 32, stride=2, downsample=conv1x1(32, 32, 2)),
    ]
    feature_layers = [ODEBlock(ODEfunc(32))]
    precapsule_layers = [norm(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=7, bias=False)]
    capsule_layers = [ PrimaryCaps(32, 8, 1, 4, 1), ConvCaps(8, 16, 3, 4, 1, 3), ConvCaps(16, 16, 1, 3, 4, 1, 3), ConvCaps(16, 10, 1, 4, 1, 3, coor_add=True, w_shared=True)]
    model = nn.Sequential(*downsampling_layers, *feature_layers, *precapsule_layers, *capsule_layers).to(device)
    """
    model = capsules(A=32, B=8, C=8, D=8, E=10, iters=args.em_iters).to(device)
    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    best_acc = test(test_loader, model, criterion, device)
    for epoch in range(1, args.epochs + 1):
        acc = train(train_loader, model, criterion, optimizer, epoch, device)

        acc /= len(train_loader)
        scheduler.step(acc)
        if epoch % args.test_intvl == 0:
            best_acc = max(best_acc, test(test_loader, model, criterion, device))
    best_acc = max(best_acc, test(test_loader, model, criterion, device))
    print('best test accuracy: {:.6f}'.format(best_acc))

    snapshot(model, args.snapshot_folder, args.epochs)

if __name__ == '__main__':
    main()
