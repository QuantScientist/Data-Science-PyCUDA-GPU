from __future__ import print_function

import argparse
import sys

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kdataset import *
from utils import *
from pycrayon import *
# Random seed

model_names = sorted(name for name in nnmodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(nnmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Ensembler')

print("Available models:" + str(model_names))

parser.add_argument('--validationRatio', type=float, default=0.90, help='test Validation Split.')
parser.add_argument('--optim', type=str, default='adam', help='Adam or SGD')
parser.add_argument('--lr_period', default=10, type=float, help='learning rate schedule restart period')
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='train batchsize')

parser.add_argument('--num_classes', type=int, default=12, help='Number of Classes in data set.')
parser.add_argument('--data_path', default='d:/db/data/seedings/train/', type=str, help='Path to train dataset')
parser.add_argument('--data_path_test', default='d:/db/data/seedings/test/', type=str, help='Path to test dataset')
parser.add_argument('--dataset', type=str, default='seeds', choices=['seeds'], help='Choose between data sets')

# parser.add_argument('--arch', metavar='ARCH', default='simple', choices=model_names)
parser.add_argument('--imgDim', default=3, type=int, help='number of Image input dimensions')
parser.add_argument('--img_scale', default=224, type=int, help='Image scaling dimensions')
parser.add_argument('--base_factor', default=20, type=int, help='SENet base factor')

parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--current_time', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    help='Current time.')

parser.add_argument('--lr', '--learning-rate', type=float, default=0.0005, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.95, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
# parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

# Checkpoints
parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--save_path_model', type=str, default='./log/', help='Folder to save checkpoints and log.')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
# random seed
parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')

# Random Erasing
from ktransforms import *
parser.add_argument('--p', default=0.32, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

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


def train(train_loader, model, criterion, optimizer, args):
    if args.use_cuda:
        model.cuda()
        criterion.cuda()

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

        if args.use_cuda:
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
        # if i % args.print_freq == 0:
        if i % 400 == 0:
            print('TRAIN: LOSS-->{loss.val:.4f} ({loss.avg:.4f})\t' 'ACC-->{acc.val:.3f}% ({acc.avg:.3f}%)'.format(loss=losses, acc=acc))
            if use_tensorboard:
                exp.add_scalar_value('tr_epoch_loss', losses.avg, step=epoch)
                exp.add_scalar_value('tr_epoch_acc', acc.avg, step=epoch)

    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))

def validate(val_loader, model, criterion, args):
    if args.use_cuda:
        model.cuda()
        criterion.cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

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

        if i % 400 == 0:
            print('VAL:   LOSS--> {loss.val:.4f} ({loss.avg:.4f})\t''ACC-->{acc.val:.3f} ({acc.avg:.3f})'.format(
                loss=losses, acc=acc))

        if i % 50 == 0:
            if use_tensorboard:
                exp.add_scalar_value('val_epoch_loss', losses.avg, step=epoch)
                exp.add_scalar_value('val_epoch_acc', acc.avg, step=epoch)

    print('FINAL ACC: {acc.avg:.3f}'.format(acc=acc))
    return float('{loss.avg:.4f}'.format(loss=losses)), float('{acc.avg:.4f}'.format(acc=acc))

# def train(train_loader, model, epoch, optimizer):
#     model.train()
#
#     for batch_idx, (data, target) in ((enumerate(train_loader))):
#         correct = 0
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         pred = output.data.max(1)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()
#         accuracy = 100. * correct / len(data)
#         optimizer.step()
#         if batch_idx % 200 == 0:
#             print('TRAIN: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.3f}%)'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.data[0],
#                 correct, len(data),
#                 accuracy))
#
#
# def test(test_loader, model, epoch):
#     #     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in (test_loader):
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += criterion(output, target).data[0]
#         pred = output.data.max(1)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()
#
#     test_loss = test_loss
#     test_loss /= len(test_loader)  # loss function already averages over batch size
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print('\nVAL: Average loss: {:.6f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         accuracy))
#
#     return test_loss, accuracy


def loadDB(args):
    # Data
    print('==> Preparing dataset %s' % args.dataset)

    classes, class_to_idx, num_to_class, df = find_classes(args.data_path)
    print('Çlasses {}'.format(classes))

    train_data = df.sample(frac=args.validationRatio)
    valid_data = df[~df['file'].isin(train_data['file'])]

    train_set = SeedDataset(train_data, args.data_path, transform=train_trans)
    valid_set = SeedDataset(valid_data, args.data_path, transform=valid_trans)

    t_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    v_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    dataset_sizes = {
        'train': len(t_loader.dataset),
        'valid': len(v_loader.dataset)
    }
    print(dataset_sizes)
    print('#Classes: {}'.format(len(classes)))
    args.num_classes = len(classes)
    args.imgDim = 3

    return t_loader, v_loader, train_set, valid_set, classes, class_to_idx, num_to_class, df


# train_trans = transforms.Compose([
#     transforms.RandomSizedCrop(args.img_scale),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],),
#     RandomErasing(),
# ])

## Augmentation + Normalization for full training
train_trans = transforms.Compose([
    transforms.RandomSizedCrop(args.img_scale),
    PowerPIL(),
    transforms.ToTensor(),
    RandomErasing(),
    normalize_img,
])

## Normalization only for validation and test
valid_trans = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(args.img_scale),
    transforms.ToTensor(),
    normalize_img
])
# valid_trans = transforms.Compose([
#     transforms.Scale(256),
#     transforms.CenterCrop(args.img_scale),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     #RandomErasing(),
# ])
# valid_trans=ds_transform_raw

test_trans = valid_trans

def testImageLoader(image_name):
    """load image, returns cuda tensor"""
#     image = Image.open(image_name)
    image = Image.open(image_name).convert('RGB')
    image = test_trans(image)
#     image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    if args.use_cuda:
        image.cuda()
    return image


def testModel(test_dir, local_model, sample_submission):
    # print ('Testing model: {}'.format(str(local_model)))
    if args.use_cuda:
        local_model.cuda()
    local_model.eval()

    columns = ['file', 'species']
    df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
    #     df_pred.species.astype(int)
    for index, row in (sample_submission.iterrows()):
        #         for file in os.listdir(test_dir):
        currImage = os.path.join(test_dir, row['file'])
        if os.path.isfile(currImage):
            X_tensor_test = testImageLoader(currImage)
            #             print (type(X_tensor_test))
            if args.use_cuda:
                X_tensor_test = Variable(X_tensor_test.cuda())
            else:
                X_tensor_test = Variable(X_tensor_test)

                # get the index of the max log-probability
            predicted_val = (local_model(X_tensor_test)).data.max(1)[1]  # get the index of the max log-probability
            #             predicted_val = predicted_val.data.max(1, keepdim=True)[1]
            p_test = (predicted_val.cpu().numpy().item())
            df_pred = df_pred.append({'file': row['file'], 'species': num_to_class[int(p_test)]}, ignore_index=True)

    return df_pred

if __name__ == '__main__':

    # tensorboad
    use_tensorboard = True
    # use_tensorboard = True and CrayonClient is not None

    if use_tensorboard == True:
        cc = CrayonClient(hostname='http://192.168.0.3')
        cc.remove_all_experiments()


    trainloader, valloader, trainset, valset, classes, class_to_idx, num_to_class, df = loadDB(args)
    models = ['wrn']
    for i in range (1,5):
        for m in models:
            runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fixSeed(args)
            model = selectModel(args, m)
            recorder = RecorderMeter(args.epochs)  # epoc is updated
            model_name = (type(model).__name__)

            exp_name = datetime.datetime.now().strftime(model_name + '_' + args.dataset + '_%Y-%m-%d_%H-%M-%S')
            if use_tensorboard == True:
                exp = cc.create_experiment(exp_name)

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
            print_log("Available models:" + str(model_names), log)
            print_log("=> Final model name '{}'".format(model_name), log)
            # print_log("=> Full model '{}'".format(model), log)
            # model = torch.nn.DataParallel(model).cuda()
            model.cuda()
            cudnn.benchmark = True
            print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
            print('Batch size : {}'.format(args.batch_size))

            criterion = torch.nn.CrossEntropyLoss()  # multi class
            # optimizer = torch.optim.Adam(model.parameters(), args.lr)  # L2 regularization
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005 * 2 * 2)


            # recorder = RecorderMeter(args.epochs)  # epoc is updated

            for epoch in tqdm(range(args.start_epoch, args.epochs)):
                train_result, accuracy_tr = train(trainloader, model, criterion, optimizer, args)
                # evaluate on validation set
                val_result, accuracy_val = validate(valloader, model, criterion,args)

                # train(trainloader, model, epoch, optimizer)
                # test_loss, accuracy_val = test(valloader, model, epoch)

                recorder.update(epoch, train_result, accuracy_tr, val_result, accuracy_val)
                mPath = args.save_path_model + '/'
                if not os.path.isdir(mPath):
                    os.makedirs(mPath)
                recorder.plot_curve(os.path.join(mPath, model_name + '_' + runId + '.png'), args, model)


                if (float(accuracy_val) > float(85.0)):
                    print ("*** EARLY STOPPING ***")
                    s_submission = pd.read_csv(args.data_path + 'sample_submission.csv')
                    s_submission.columns = ['file', 'species']
                    df_pred = testModel(args.data_path_test, model, s_submission)

                    pre = args.save_path_model + '/' + '/pth/'
                    if not os.path.isdir(pre):
                        os.makedirs(pre)
                    fName = pre + str(accuracy_val)
                    torch.save(model.state_dict(), fName + '_cnn.pth')
                    csv_path = str(fName + '_submission.csv')
                    df_pred.to_csv(csv_path, columns=('file', 'species'), index=None)
                    # df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
                    print(csv_path)
