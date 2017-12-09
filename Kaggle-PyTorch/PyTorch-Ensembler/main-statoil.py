from __future__ import print_function

import argparse
import sys

import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils import *
from losses import Eve

model_names = sorted(name for name in nnmodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(nnmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 and 100 Training')

print("Available models:" + str(model_names))

parser.add_argument('--validationRatio', type=float, default=0.11, help='test Validation Split.')
parser.add_argument('--optim', type=str, default='adam', help='Adam or SGD')
parser.add_argument('--lr_period', default=10, type=float, help='learning rate schedule restart period')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='train batchsize')

parser.add_argument('--num_classes', type=int, default=1, help='Number of Classes in data set.')
parser.add_argument('--data_path', default='d:/db/data/ice/', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, default='statoil', choices=['statoil', 'statoil'],
                    help='Choose between Statoil.')

# parser.add_argument('--num_classes', type=int, default=10, help='Number of Classes in data set.')
# parser.add_argument('--data_path', default='d:/db/data/cifar10/', type=str, help='Path to dataset')
# parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'Iceberg'],help='Choose between Cifar10/100 and ImageNet.')


# parser.add_argument('--arch', metavar='ARCH', default='senet', choices=model_names)
parser.add_argument('--imgDim', default=2, type=int, help='number of Image input dimensions')
parser.add_argument('--base_factor', default=32, type=int, help='SENet base factor')

parser.add_argument('--epochs', type=int, default=66, help='Number of epochs to train.')
parser.add_argument('--current_time', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    help='Current time.')

# parser.add_argument('--learning_rate', type=float, default=0.0005, help='The Learning Rate.')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.0005, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.95, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--save_path_model', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
# random seed
parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

# Use CUDA
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
use_cuda = args.use_cuda

if args.manualSeed is None:
    args.manualSeed = 999
fixSeed(args)


def BinaryTrainAndValidate(model, criterion, optimizer, runId, debug=False):
    if args.use_cuda:
        model.cuda()
        criterion.cuda()
    all_losses = []
    val_losses = []

    for epoch in tqdm(range(args.epochs)):
        adjust_learning_rate(optimizer,epoch, args)
        model.train()
        tqdm.write('\n==>>Epoch=[{:03d}/{:03d}]], {:s}, LR=[{}], Batch=[{}]'.format(epoch, args.epochs, time_string(),
                                                                                    state['lr'],
                                                                                    args.batch_size) + ' [Model={}]'.format(
            (type(model).__name__), ), log)

        running_loss = 0.0
        # for i, row_data in tqdm (enumerate(trainloader, 1)):
        for i, row_data in (enumerate(trainloader, 1)):
            img, label = row_data
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))  # On GPU
            else:
                img, label = Variable(img), Variable(label)  # RuntimeError: expected CPU tensor (got CUDA tensor)

            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.data[0] * label.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(running_loss / (args.batch_size * i))
        predicted_tr = (model(img).data > 0.5).float()
        accuracy_tr = (predicted_tr == label.data).float().mean() * 100

        # model.eval()
        eval_loss = 0
        for row_data in testloader:
            img, label = row_data
            if use_cuda:
                img, label = Variable(img.cuda(async=True), volatile=True), Variable(label.cuda(async=True),
                                                                                     volatile=True)  # On GPU
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)

        val_losses.append(eval_loss / (len(testset)))

        predicted_val = (model(img).data > 0.5).float()
        # predictions_val = predicted_val.cpu().numpy()
        accuracy_val = (predicted_val == label.data).float().mean() * 100

        if debug is True:
            tqdm.write('-->LOSS T/V:[{:.6f}/{:.6f}%], ACC T/V:[{:.6f}/{:.6f}%]'.format(running_loss / (len(trainset)),
                                                                                       eval_loss / (len(testset)),
                                                                                       accuracy_tr, accuracy_val))

        val_result = float('{:.6f}'.format(eval_loss / (len(testset))))
        train_result = float('{:.6f}'.format(running_loss / (len(trainset))))

        recorder.update(epoch, train_result, accuracy_tr, val_result, accuracy_val)
        mPath = args.save_path_model + '/'
        if not os.path.isdir(mPath):
            os.makedirs(mPath)
        recorder.plot_curve(os.path.join(mPath, model_name + '_' + runId + '.png'), args, model)
        logger.append([state['lr'], train_result, val_result, accuracy_tr, accuracy_val])

        if (float(val_result) < float(0.175) and float(train_result) < float(0.175)):
            print_log("=>>EARLY STOPPING", log)
            df_pred = BinaryInference(model, args)
            savePred(df_pred, model, str(val_result) + '_' + str(epoch), train_result, args.save_path_model)
            # break
            continue
            # adjust_learning_rate(optimizer, epoch)

    tqdm.write('TRAIN Loss: {:.6f}'.format(running_loss / (len(trainset))), log)
    tqdm.write('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(testset))), log)
    val_result = '{:.6f}'.format(eval_loss / (len(testset)))
    train_result = '{:.6f}'.format(running_loss / (len(trainset)))

    return val_result, train_result


def loadDB(args):
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'statoil':
        args.num_classes = 1
        args.imgDim = 2
        trainloader, testloader, trainset, testset = getStatoilTrainValLoaders(args)

    return trainloader, testloader, trainset, testset


if __name__ == '__main__':

    # ensembleVer2('./pth_old/raw/iceResNet/', './pth_old/ens2/ens_ice800files.csv')
    # MinMaxBestBaseStacking('./pth_old/2020/', './pth_old/2020/0.1339.csv','./pth_old/2020/final_mix_900_files_base01344.csv')
    # ensembleVer2('./log/statoil/IceResNet/pth', './ens_ice_98989898989898989.csv')
    # ensembleVer2('./log/DenseNet/pth/', './pth_old/ens2/ens_densnet_1_hours.csv')

    # vis = visdom.Visdom(port=6006)
    trainloader, testloader, trainset, testset = loadDB(args)
    # for i in tqdm(range(0, 51)):
    for i in range(0, 100):
        models = ['senet','resnext','simple']
        for m in models:
            runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fixSeed(args)
            model = selectModel(args, m)
            model_name = (type(model).__name__)
            # if model_name =='NoneType':
            #     EXIT
            mPath = args.save_path + '/' + args.dataset + '/' + model_name + '/'
            args.save_path_model = mPath
            if not os.path.isdir(args.save_path_model):
                mkdir_p(args.save_path_model)
            log = open(os.path.join(args.save_path_model, 'log_seed_{}_{}.txt'.format(args.manualSeed, runId)), 'w')
            print_log('Save path : {}'.format(args.save_path_model), log)
            print_log(state, log)
            print_log("Random Seed: {}".format(args.manualSeed), log)
            print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
            print_log("torch  version : {}".format(torch.__version__), log)
            print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
            print_log("LR :" + str(args.lr), log)
            print_log("Available models:" + str(model_names), log)
            print_log("=> Final model name '{}'".format(model_name), log)
            # print_log("=> Full model '{}'".format(model), log)
            # model = torch.nn.DataParallel(model).cuda()
            model.cuda()
            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

            if args.num_classes == 1:
                criterion = torch.nn.BCELoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()
            if args.optim is 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.lr)  # L2 regularization
            else:
                optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=state['momentum'],
                                            weight_decay=state['weight_decay'], nesterov=True)

            # print_log("=> Criterion '{}'".format(str(criterion)), log)
            # print_log("=> optimizer '{}'".format(str(optimizer)), log)

            title = model_name
            logger = Logger(os.path.join(args.save_path_model, runId + '_log.txt'), title=title)
            logger.set_names(['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc.', 'ValidAcc.'])
            recorder = RecorderMeter(args.epochs)  # epoc is updated

            val_result, train_result = BinaryTrainAndValidate(model, criterion, optimizer, runId, debug=True)
            if (float(val_result) < float(0.165) and float(train_result) < float(0.165)):
                df_pred = BinaryInference(model)
                savePred(df_pred, model, val_result, train_result, args.save_path_model)

            logger.close()
            logger.plot()
