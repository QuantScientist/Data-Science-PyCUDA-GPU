from __future__ import print_function

import time

import matplotlib
import torch.nn as nn
import torch.nn.init as init

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from torch.utils.data import TensorDataset

import torch.nn.parallel
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torchvision.transforms import *
import argparse
import csv
import os
import os.path
import shutil
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import datetime
import random

from utils import *

# model_names = sorted(name for name in nnmodels.__dict__ if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch SENet for TF commands')

parser.add_argument('--dataset', type=str, default='tf', choices=['tf'], help='Choose between data sets')
parser.add_argument('--train_path', default='d:/db/data/tf/2018/train', help='path to the train data folder')
parser.add_argument('--test_path', default='d:/db/data/tf/2018/test', help='path to the test data folder')
parser.add_argument('--valid_path', default='d:/db/data/tf/2018/valid', help='path to the valid data folder')
parser.add_argument('--test_audio', default='d:/db/data/tf/test/audio/', help='path to the valid data folder')

parser.add_argument('--save_path', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--save_path_model', type=str, default='./log/', help='Folder to save checkpoints and log.')

parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0005 * 2 * 2 , type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=400, type=int, metavar='N', help='print frequency')
parser.add_argument('--test', default=False, help='evaluate model on test set')

parser.add_argument('--validationRatio', type=float, default=0.11, help='test Validation Split.')
parser.add_argument('--optim', type=str, default='adam', help='Adam or SGD')
parser.add_argument('--imgDim', default=1, type=int, help='number of Image input dimensions')
parser.add_argument('--img_scale', default=224, type=int, help='Image scaling dimensions')
parser.add_argument('--base_factor', default=20, type=int, help='SENet base factor')

parser.add_argument('--current_time', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),help='Current time.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
# random seed
parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')

parser.add_argument('--num_classes', type=int, default=12, help='Number of Classes in data set.')


# feature extraction options
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')
parser.add_argument('--normalize', default=True, help='boolean, wheather or not to normalize the spect')

import librosa
import numpy as np

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

def fixSeed(args):
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)

# Use CUDA
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
use_cuda = args.use_cuda

if args.manualSeed is None:
    args.manualSeed = 999
fixSeed(args)



import librosa
import numpy as np
import librosa
import numpy as np

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    num_to_class = dict(zip(range(len(classes)), classes))
    return classes, class_to_idx, num_to_class




def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class TFAudioDataSet(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx, idx_to_class = find_classes(root)
        print ('Çlasses {}'.format(classes))
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)

best_prec1 = 0


def testAudioLoader(image_name):
    """load image, returns cuda tensor"""
#     image = Image.open(image_name)
    image = spect_loader(image_name, args.window_size, args.window_stride, args.window_type, args.normalize, max_len=101)
#     image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    if args.use_cuda:
        image.cuda()
    return image

def testModel(test_dir, local_model, sample_submission):
    print ('Testing model: {}'.format(str(local_model)))

    classes, class_to_idx, idx_to_class = find_classes(args.train_path)

    if args.use_cuda:
        local_model.cuda()
    local_model.eval()

    columns = ['fname', 'label']
    df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
    #     df_pred.species.astype(int)
    for index, row in (sample_submission.iterrows()):
        #         for file in os.listdir(test_dir):
        currImage = os.path.join(test_dir, row['fname'])
        if os.path.isfile(currImage):
            print (currImage)
            X_tensor_test = testAudioLoader(currImage)
            #             print (type(X_tensor_test))
            if args.use_cuda:
                X_tensor_test = Variable(X_tensor_test.cuda())
            else:
                X_tensor_test = Variable(X_tensor_test)
            predicted_val = (local_model(X_tensor_test)).data.max(1)[1]  # get the index of the max log-probability
            p_test = (predicted_val.cpu().numpy().item())
            df_pred = df_pred.append({'fname': row['fname'], 'label': idx_to_class[int(p_test)]}, ignore_index=True)

    print('Testing model done: {}'.format(str(df_pred.shape)))
    return df_pred

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            images, target = images.cuda(), target.cuda()
            images, target = Variable(images), Variable(target)

        # compute y_pred
        y_pred = model(images)
        loss = criterion(y_pred, target)

        # measure accuracy and record loss
        prec1, prec1 = accuracy(y_pred.data, target.data, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('TRAIN: LOSS-->{loss.val:.4f} ({loss.avg:.4f})\t' 'ACC-->{acc.val:.3f}% ({acc.avg:.3f}%)'.format(loss=losses, acc=acc))

    return  float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    # model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):

        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images, volatile=True), Variable(labels)

        # compute y_pred
        y_pred = model(images)
        loss = criterion(y_pred, labels)

        # measure accuracy and record loss
        prec1, temp_var = accuracy(y_pred.data, labels.data, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        acc.update(prec1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('VAL:   LOSS--> {loss.val:.4f} ({loss.avg:.4f})\t''ACC-->{acc.val:.3f} ({acc.avg:.3f})'.format(loss=losses, acc=acc))
    print(' * Accuracy {acc.avg:.3f}'.format(acc=acc))
    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))


def save_checkpoint(state, is_best, acc):
    filename= args.save_path_model + '/' + str(acc) + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


import errno
import time

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(
            self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain:
            return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:
            return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path, args, model):
        title = 'PyTorch Model:' + str((type(model).__name__)).upper() + ', DataSet:' + str(args.dataset).upper() + ',' \
                + 'Params: %.2fM' % (
            sum(p.numel() for p in model.parameters()) / 1000000.0) + ', Seed: %.2f' % args.manualSeed
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 1.0)
        interval_y = 0.05 / 3.0
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 1.0 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=18)
        plt.xlabel('EPOCH', fontsize=16)
        plt.ylabel('LOSS/ACC', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0] / 100.0
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='tr-accuracy/100', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1] / 100.0
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='val-accuracy/100', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='r', linestyle=':', label='tr-loss', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='b', linestyle=':', label='val-loss', lw=4)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def main():
    global args, best_prec1

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    # fixSeed(args)
    models = ['senet']

    for m in models:
        model = selectModel(args, m)
        # model = models.__dict__[args.arch]()
        model_name = (type(model).__name__)

        mPath = args.save_path + '/' + args.dataset + '/' + model_name + '/'
        args.save_path_model = mPath
        if not os.path.isdir(args.save_path_model):
            mkdir_p(args.save_path_model)

        print("Ensemble with model {}:".format(model_name))
        print('Save path : {}'.format(args.save_path_model))
        print(state)
        print("Random Seed: {}".format(args.manualSeed))
        import sys
        print("python version : {}".format(sys.version.replace('\n', ' ')))
        print("torch  version : {}".format(torch.__version__))
        print("cudnn  version : {}".format(torch.backends.cudnn.version()))
        print("=> Final model name '{}'".format(model_name))
        # print_log("=> Full model '{}'".format(model), log)
        # model = torch.nn.DataParallel(model).cuda()
        model.cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        print('Batch size : {}'.format(args.batch_size))

        if args.use_cuda:
            model.cuda()
            # model = torch.nn.DataParallel(model).cuda()

        cudnn.benchmark = True
        # Data loading code
        train_dataset = TFAudioDataSet(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                                       window_type=args.window_type, normalize=args.normalize)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=False, sampler=None)
        valid_dataset = TFAudioDataSet(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                                       window_type=args.window_type, normalize=args.normalize)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=None, num_workers=0,
                                                   pin_memory=False, sampler=None)
        test_dataset = TFAudioDataSet(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                                      window_type=args.window_type, normalize=args.normalize)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=None,
                                                  num_workers=0,
                                                  pin_memory=False, sampler=None)

        # define loss function (criterion)
        criterion = nn.CrossEntropyLoss()
        if args.use_cuda:
            criterion.cuda()

        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        recorder = RecorderMeter(args.epochs)  # epoc is updated
        runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if args.test:
            print("Testing the model and generating  output csv for submission")
            s_submission = pd.read_csv('tf-sample-submission.csv')
            s_submission.columns = ['fname', 'label']

            checkpoint = torch.load('./log/tf/ResNeXt/checkpoint.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])

            df_pred= testModel (args.test_audio, model, s_submission)
            pre = args.save_path_model + '/' + '/pth/'
            if not os.path.isdir(pre):
                os.makedirs(pre)
            fName = pre + str('.83')
            # torch.save(model.state_dict(), fName + '_cnn.pth')
            csv_path = str(fName + '_submission.csv')
            df_pred.to_csv(csv_path, columns=('fname', 'label'), index=None)
            print(csv_path)

            return


            # for epoch in range(args.start_epoch, args.epochs):
        for epoch in tqdm(range(args.start_epoch, args.epochs)):
            adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            tqdm.write('\n==>>Epoch=[{:03d}/{:03d}]], LR=[{}], Batch=[{}]'.format(epoch, args.epochs,
                                                                                        state['lr'],
                args.batch_size) + ' [Model={}]'.format(
                (type(model).__name__), ))

            train_result, accuracy_tr=train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_result, accuracy_val = validate(val_loader, model, criterion)

            recorder.update(epoch, train_result, accuracy_tr, val_result, accuracy_val)
            mPath = args.save_path_model + '/'
            if not os.path.isdir(mPath):
                os.makedirs(mPath)
            recorder.plot_curve(os.path.join(mPath, model_name + '_' + runId + '.png'), args, model)

            # remember best Accuracy and save checkpoint
            is_best = accuracy_val > best_prec1
            best_prec1 = max(accuracy_val, best_prec1)

            if float(accuracy_val) > float(70.0):
                print("*** EARLY STOPPING ***")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, best_prec1)

                print("Testing the model and generating  output csv for submission")
                s_submission = pd.read_csv('tf-sample-submission.csv')
                s_submission.columns = ['fname', 'label']
                df_pred = testModel(args.test_audio, model, s_submission)
                pre = args.save_path_model + '/' + '/pth/'
                if not os.path.isdir(pre):
                    os.makedirs(pre)
                fName = pre + str('.83')
                # torch.save(model.state_dict(), fName + '_cnn.pth')
                csv_path = str(fName + '_submission.csv')
                df_pred.to_csv(csv_path, columns=('fname', 'label'), index=None)
                print(csv_path)

        test_loss, test_acc = validate(test_loader, model, criterion)
        print('Test: {}, {}'.format(test_loss, test_acc))


if __name__ == '__main__':
    main()
