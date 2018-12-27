from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
from tqdm import tqdm
import PIL.Image as Image

#torch.cuda.set_device(0)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lossfile', type=str, default='output_rec.csv', metavar='N',
                    help='record file')
parser.add_argument('--sgd', type=int, default=0, metavar='N',
                    help='sgd or adadelta')
parser.add_argument('--model_name', type=str, default='model_100.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--accfile', type=str, default='acc.csv', metavar='N',
                    help='accuracy file')

args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()
model.cuda()

model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.sgd == 0:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
else:
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)

loss_file = open(args.lossfile+'.csv', "w")
acc_file = open(args.accfile+'.csv', "w")

def train(epoch):
    model.train()
    train_loss_rec = 0
    batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss_rec += loss.data.item()
        batches = batch_idx
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    loss_file.write("%s, %.6f\n" % (0, train_loss_rec/batches))

    correct = 0
    for data, target in train_loader:
        with torch.no_grad():
            data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    correct = 100. * correct / len(train_loader.dataset)
    acc_file.write("%s, %.4f\n" % (0, correct))
    print('Training accuracy:{:.4f}'.format(correct))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    loss_file.write("%s, %.4f\n" % (1, validation_loss))
    acc_file.write("%s, %.4f\n" % (1, 100. * correct / len(val_loader.dataset)))

if args.sgd == 0:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.4)

for epoch in range(1, args.epochs + 1):
    if args.sgd == 0:
        scheduler.step()
    train(epoch)
    validation()
    model_file = '/scratch/lc3909/assign2/model_' + args.model_name + '_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')

test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")
for f in tqdm(os.listdir(test_dir)):
    if 'ppm' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        with torch.no_grad():
             data = Variable(data)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]

        file_id = f[0:5]
        output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2017/')