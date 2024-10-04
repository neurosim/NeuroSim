import argparse
import os
import time
from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from cifar import dataset
from cifar import model
from utee import hook
#from IPython import embed
from datetime import datetime
from subprocess import call
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=257, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--wl_weight', default=2)
parser.add_argument('--wl_grad', default=8)
parser.add_argument('--wl_activate', default=8)
parser.add_argument('--wl_error', default=8)
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

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

model_path = './log/default/batch_size=200/decreasing_lr=200,250/grad_scale=8/seed=117/type=cifar10/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=2/latest.pth'


# data loader and model
assert args.type in ['cifar10', 'cifar100'], args.type
train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)
model = model.cifar10(args = args, logger=logger, pretrained = model_path)
print(args.cuda)
if args.cuda:
    model.cuda()

best_acc, old_file = 0, None
t_begin = time.time()


# ready to go
model.eval()
test_loss = 0
correct = 0
trained_with_quantization = True
# for data, target in test_loader:
for i, (data, target) in enumerate(test_loader):
    if i==0:
        hook_handle_list = hook.hardware_evaluation(model,args.wl_weight,args.wl_activate)
    indx_target = target.clone()
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    if i==0:
        hook.remove_hook_list(hook_handle_list)


test_loss = test_loss / len(test_loader)  # average over number of mini-batch
acc = 100. * correct / len(test_loader.dataset)
logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset), acc))

call(["/bin/bash", "./layer_record/trace_command.sh"])