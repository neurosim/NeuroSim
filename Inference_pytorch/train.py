import argparse
import os
import time
from utee import misc
import torch
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from models import dataset
from utee import wage_util
from datetime import datetime
from utee import wage_quantizer


parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', type = int, default=8)
parser.add_argument('--wl_grad', type = int, default=8)
parser.add_argument('--wl_activate', type = int, default=8)
parser.add_argument('--wl_error', type = int, default=8)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--ADCprecision', default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', default=4, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', default=0, help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', default=0, help='retention time')
parser.add_argument('--v', default=0, help='drift coefficient')
parser.add_argument('--detect', default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', default=0, help='drift target for fixed-direction drift')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs'])
misc.logger.init(args.logdir, 'train_log_' +current_time)
logger = misc.logger.info

# logger
misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader and model
assert args.dataset in ['cifar10', 'cifar100', 'imagenet'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'imagenet':
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
else:
    raise ValueError("Unknown dataset type")
    
assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from models import VGG
    model = VGG.vgg8(args = args, logger=logger)
    criterion = wage_util.SSE()
elif args.model == 'DenseNet40':
    from models import DenseNet
    model = DenseNet.densenet40(args = args, logger=logger)
    criterion = wage_util.SSE()
elif args.model == 'ResNet18':
    from models import ResNet
    model = ResNet.resnet18(args = args, logger=logger)
    criterion = torch.nn.CrossEntropyLoss()
else:
    raise ValueError("Unknown model type")

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()
grad_scale = args.grad_scale

try:
    # ready to go
    for epoch in range(args.epochs):
        model.train()

        if epoch in decreasing_lr:
             grad_scale = grad_scale / 8.0

        logger("training phase")
        for batch_idx, (data, target) in enumerate(train_loader):
            indx_target = target.clone()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)

            loss.backward()
            
            if args.mode == 'WAGE':
                for name, param in list(model.named_parameters())[::-1]:
                    param.grad.data = wage_quantizer.QG(param.grad.data,args.wl_grad,grad_scale)

            optimizer.step()
            
            if args.mode == 'WAGE':
                for name, param in list(model.named_parameters())[::-1]:
                    param.data = wage_quantizer.C(param.data, args.wl_weight)

            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) * 1.0 / len(data)
                logger('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    loss.data, acc, optimizer.param_groups[0]['lr']))

        elapse_time = time.time() - t_begin
        speed_epoch = elapse_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * args.epochs - elapse_time
        logger("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))

        misc.model_save(model, os.path.join(args.logdir, 'latest.pth'))

        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            logger("testing phase")
            for i, (data, target) in enumerate(test_loader):
                indx_target = target.clone()
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                    output = model(data)
                    test_loss_i = criterion(output, target)
                    test_loss += test_loss_i.data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(indx_target).sum()

            test_loss = test_loss / len(test_loader) # average over number of mini-batch
            acc = 100. * correct / len(test_loader.dataset)
            logger('\tEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, test_loss, correct, len(test_loader.dataset), acc))

            if acc > best_acc:
                new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                misc.model_save(model, new_file, old_file=old_file, verbose=True)
                best_acc = acc
                old_file = new_file

except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    logger("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))


