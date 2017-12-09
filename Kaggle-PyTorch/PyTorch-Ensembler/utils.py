'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import print_function, absolute_import

import errno
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

import nnmodels as nnmodels

from os import listdir
import sys

# __all__ = ['Logger', 'LoggerMonitor', 'savefig']
# __all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'accuracy']

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def savefig(fname, dpi=None):
    dpi = 500 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        if name in ['Train Acc.', 'Valid Acc.']:
            plt.plot(x, 100 - np.asarray(numbers[name], dtype='float'))
        else:
            plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''

    def __init__(self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.plot()
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        legend_text = ['WRN-28-10+Ours (error 17.65%)', 'WRN-28-10 (error 18.68%)']
        plt.legend(legend_text, loc=0)
        plt.ylabel('test error (%)')
        plt.xlabel('epoch')
        plt.grid(True)


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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


class TrainningValidationSplitDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(TrainningValidationSplitDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]


def trainTestSplit(dataset, val_share):
    val_offset = int(len(dataset) * (1 - val_share))
    # print("Offest:" + str(val_offset))
    return TrainningValidationSplitDataset(dataset, 0, val_offset), TrainningValidationSplitDataset(dataset, val_offset,
                                                                                                    len(
                                                                                                        dataset) - val_offset)


def createNewDir(BASE_FOLDER):
    parquet_dir = os.path.join(BASE_FOLDER, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(parquet_dir)
    return parquet_dir


def savePred(df_pred, local_model, val_score, train_score, save_path):
    pre = save_path + '/' + '/pth/'
    if not os.path.isdir(pre):
        os.makedirs(pre)
    fName = pre + str(val_score) + '_' + str(train_score)
    torch.save(local_model.state_dict(), fName + '_cnn.pth')
    csv_path = str(fName + '_submission.csv')
    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
    print(csv_path)


def MinMaxBestBaseStacking(input_folder, best_base, output_path):
    sub_base = pd.read_csv(best_base)
    all_files = os.listdir(input_folder)

    # Read and concatenate submissions
    outs = [pd.read_csv(os.path.join(input_folder, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)

    # get the data fields ready for stacking
    concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)
    concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)
    concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)
    concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)

    # set up cutoff threshold for lower and upper bounds, easy to twist
    cutoff_lo = 0.67
    cutoff_hi = 0.33

    concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
    concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:6] > cutoff_lo, axis=1),
                                        concat_sub['is_iceberg_max'],
                                        np.where(np.all(concat_sub.iloc[:, 1:6] < cutoff_hi, axis=1),
                                                 concat_sub['is_iceberg_min'],
                                                 concat_sub['is_iceberg_base']))
    concat_sub[['id', 'is_iceberg']].to_csv(output_path,
                                            index=False, float_format='%.12f')


def ensembleVer2(input_folder, output_path):
    print('Out:' + output_path)
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    model_scores = []
    for i, csv in enumerate(csv_files):
        df = pd.read_csv(os.path.join(input_folder, csv), index_col=0)
        if i == 0:
            index = df.index
        else:
            assert index.equals(df.index), "Indices of one or more files do not match!"
        model_scores.append(df)
    print("Read %d files. Averaging..." % len(model_scores))

    # print(model_scores)
    concat_scores = pd.concat(model_scores)
    print(concat_scores.head())
    concat_scores['is_iceberg'] = concat_scores['is_iceberg'].astype(np.float32)

    averaged_scores = concat_scores.groupby(level=0).mean()
    assert averaged_scores.shape[0] == len(list(index)), "Something went wrong when concatenating/averaging!"
    averaged_scores = averaged_scores.reindex(index)

    stacked_1 = pd.read_csv('statoil-submission-template.csv')  # for the header
    print(stacked_1.shape)
    sub = pd.DataFrame()
    sub['id'] = stacked_1['id']

    sub['is_iceberg'] = np.exp(np.mean(
        [
            averaged_scores['is_iceberg'].apply(lambda x: np.log(x))
        ], axis=0))

    print(sub.shape)
    sub.to_csv(output_path, index=False, float_format='%.9f')
    print("Averaged scores saved to %s" % output_path)


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def XnumpyToTensor(x_data_np, args):
    x_data_np = np.array(x_data_np, dtype=np.float32)

    if args.use_cuda:
        X_tensor = (torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
    else:
        X_tensor = (torch.from_numpy(x_data_np))  # Note the conversion for pytorch

    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np, args):
    y_data_np = y_data_np.reshape((y_data_np.shape[0], 1))  # Must be reshaped for PyTorch!

    if args.use_cuda:
        #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor).cuda()  # BCEloss requires Float
    else:
        #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #
        Y_tensor = (torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float

    return Y_tensor


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
        title = 'PyTorch-Ensembler:' + str((type(model).__name__)).upper() + ',LR:' + str(args.lr) +  ',DataSet:' + str(args.dataset).upper() + ',' + '\n'\
                + ',Params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0) + ',Seed: %.2f' % args.manualSeed + \
                ",Torch: {}".format(torch.__version__) + ", Batch:{}".format(args.batch_size)

        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 6
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


def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


import math


# https://github.com/gngdb/pytorch-cifar-sgdr/blob/master/main.py
def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx / restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi * (batch_idx / restart_period)
    return 0.5 * (1.0 + math.cos(radians))


# def adjust_learning_rate(optimizer, epoch):
#     global lr
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = lr * (0.01 ** (epoch // 10))
#     for param_group in optimizer.state_dict()['param_groups']:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 after 20 and 40  and 60 epochs"""
    # global lr
    lr = args.lr * (0.1 ** (epoch // 25)) * (0.1 ** (epoch //  45)) * (0.1 ** (epoch //  55))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fixSeed(args):
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        torch.cuda.manual_seed_all(args.manualSeed)


def getStatoilTrainValLoaders(args):
    fixSeed(args)
    local_data = pd.read_json(args.data_path + '/train.json')

    local_data = shuffle(local_data)  # otherwise same validation set each time!
    local_data = local_data.reindex(np.random.permutation(local_data.index))

    local_data['band_1'] = local_data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    local_data['band_2'] = local_data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    local_data['inc_angle'] = pd.to_numeric(local_data['inc_angle'], errors='coerce')
    local_data['inc_angle'].fillna(0, inplace=True)

    band_1 = np.concatenate([im for im in local_data['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in local_data['band_2']]).reshape(-1, 75, 75)
    # band_3=(band_1+band_2)/2
    local_full_img = np.stack([band_1, band_2], axis=1)

    train_imgs = XnumpyToTensor(local_full_img, args)
    train_targets = YnumpyToTensor(local_data['is_iceberg'].values, args)
    dset_train = TensorDataset(train_imgs, train_targets)

    local_train_ds, local_val_ds = trainTestSplit(dset_train, args.validationRatio)
    local_train_loader = torch.utils.data.DataLoader(local_train_ds, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
    local_val_loader = torch.utils.data.DataLoader(local_val_ds, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers)
    return local_train_loader, local_val_loader, local_train_ds, local_val_ds


def selectModel(args, m):
    model = None
    print("==> Creating model '{}'".format(m))
    if m.startswith('senet'):  # block, n_size=1, num_classes=1, num_rgb=2, base=32
        # model = nnmodels.senetXX_generic(args.num_classes, args.imgDim, args.base_factor)
        model = nnmodels.senet32_RG_1_classes(args.num_classes, args.imgDim)
        args.batch_size = 64
        args.batch_size = 64
        args.epochs = 56
        args.lr =  0.0005 # do not change !!! optimal for the Statoil data set

    if m.startswith('densenet'):
        model = nnmodels.densnetXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 64
        args.batch_size = 64
        args.epochs = 75
        args.lr = 0.05
    if m.startswith('minidensenet'):
        model = nnmodels.minidensnetXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 32
        args.batch_size = 32
        args.epochs = 76
        args.lr = 0.0005 * 2
    if m.startswith('vggnet'):
        model = nnmodels.vggnetXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 64
        args.batch_size = 64
        args.epochs = 88
        args.lr = 0.0005
    if m.startswith('resnext'):
        model = nnmodels.resnetxtXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 16
        args.batch_size = 16
        args.epochs = 66
    if m.startswith('lenet'):
        model = nnmodels.lenetXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 64
        args.batch_size = 64
        args.epochs = 88

    if m.startswith('wrn'):
        model = nnmodels.wrnXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 16
        args.batch_size = 16
        args.epochs = 56

    if m.startswith('simple'):
        model = nnmodels.simpleXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 128
        args.batch_size = 128
        args.epochs = 120

    if m.startswith('unet'):
        model = nnmodels.unetXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 64
        args.batch_size = 64
        args.epochs = 50

    if m.startswith('link'):
        model = nnmodels.linknetXX_generic(args.num_classes, args.imgDim)
        args.batch_size = 64
        args.batch_size = 64
        args.epochs = 50

    return model


def BinaryInference(local_model, args):
    local_model.eval()
    df_test_set = pd.read_json(args.data_path + '/test.json')
    df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')
    # df_test_set.head(3)
    print(df_test_set.shape)
    columns = ['id', 'is_iceberg']
    df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
    # df_pred.id.astype(int)
    for index, row in df_test_set.iterrows():
        rwo_no_id = row.drop('id')
        band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
        band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
        # band_3_test = (band_1_test + band_2_test) / 2
        full_img_test = np.stack([band_1_test, band_2_test], axis=1)

        x_data_np = np.array(full_img_test, dtype=np.float32)
        if args.use_cuda:
            X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
        else:
            X_tensor_test = Variable(torch.from_numpy(x_data_np))  # Note the conversion for pytorch

        # X_tensor_test=X_tensor_test.view(1, trainX.shape[1]) # does not work with 1d tensors
        predicted_val = (local_model(X_tensor_test).data).float()  # probabilities
        p_test = predicted_val.cpu().numpy().item()  # otherwise we get an array, we need a single float

        df_pred = df_pred.append({'id': row['id'], 'is_iceberg': p_test}, ignore_index=True)

    return df_pred


def find_classes(fullDir):
    classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    num_to_class = dict(zip(range(len(classes)), classes))

    train = []
    for index, label in enumerate(classes):
        path = fullDir + label + '/'
        for file in listdir(path):
            train.append(['{}/{}'.format(label, file), label, index])

    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])

    return classes, class_to_idx, num_to_class, df

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


import random
from math import floor

# adapted from https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/main_pytorch.py
def train_valid_split(dataset, test_size=0.25, shuffle=False, random_seed=0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.

    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    length = dataset.__len__()
    indices = list(range(1, length))

    if shuffle == True:
        random.seed(random_seed)
        random.shuffle(indices)

    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]