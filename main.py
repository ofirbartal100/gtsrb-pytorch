from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import torchattacks as ta

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
	use_gpu = False
	print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

### Data Initialization and Loading
from data import initialize_data, data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_hflip,data_vflip # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set


train_data_path = args.data + '/train_images'
val_data_path = args.data + '/val_images'
# Apply data transformations on the training images to augment dataset
train_loader = torch.utils.data.DataLoader(
   torch.utils.data.ConcatDataset([ datasets.ImageFolder(train_data_path,transform=data_transforms),
                                    datasets.ImageFolder(train_data_path,transform=data_jitter_brightness),
                                    datasets.ImageFolder(train_data_path,transform=data_jitter_hue),
                                    datasets.ImageFolder(train_data_path,transform=data_jitter_contrast),
                                    datasets.ImageFolder(train_data_path,transform=data_jitter_saturation),
                                    datasets.ImageFolder(train_data_path,transform=data_translate),
                                    datasets.ImageFolder(train_data_path,transform=data_rotate),
                                    datasets.ImageFolder(train_data_path,transform=data_hvflip),
                                    datasets.ImageFolder(train_data_path,transform=data_center),
                                    datasets.ImageFolder(train_data_path,transform=data_shear)
                                    ]), 
   batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)
   
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_data_path,transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)
   

# Neural Network and Optimizer
from model import Net
model = Net()

if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)

def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model.forward_original(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()/(args.batch_size * args.log_interval),loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

def adv_train(epoch,atk):
    p_adv = 0.5

    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()

        adv_data = atk(data,target)
        adv_mask = torch.rand(target.shape) < p_adv
        shape = list(data.shape)
        shape[0] = -1
        target_shape = list(target.shape)
        target_shape[0] = -1
        data_mix = torch.cat([data[~adv_mask].view(shape),adv_data[adv_mask].view(shape)],axis = 0)
        target_mix = torch.cat([target[~adv_mask].view(target_shape),target[adv_mask].view(target_shape)],axis = 0)

        optimizer.zero_grad()
        output = model.forward_original(data_mix)
        loss = F.nll_loss(output, target_mix)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target_mix).sum()
        training_loss += loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_mix), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()/(args.batch_size * args.log_interval),loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model.forward_original(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss,2))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    return correct / len(val_loader.dataset)

def adv_validation(atk):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        adv_data = atk(data,target)
        output = model.forward_original(adv_data)
        validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss,2))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    return correct / len(val_loader.dataset)




threat_model = Net()
threat_model.load_state_dict(torch.load('/workspace/gtsrb_pytorch/model/model_40.pth'))
threat_model.eval()
if use_gpu:
    threat_model.cuda()

atk = ta.FGSM(threat_model, eps=0.005) # adv attack
MEAN,STD = np.array([0.3337, 0.3064, 0.3171],dtype=np.float32), np.array([ 0.2672, 0.2564, 0.2629],dtype=np.float32)
atk.set_normalization_used(mean=MEAN, std=STD)

best_acc = 0.6
for epoch in range(1, args.epochs + 1):
    # train(epoch)
    adv_train(epoch,atk)
    acc = validation()
    adv_acc = adv_validation(atk)
    if acc > best_acc:
        best_acc = acc
        model_file = 'model_defended.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + f' with acc {acc}')
