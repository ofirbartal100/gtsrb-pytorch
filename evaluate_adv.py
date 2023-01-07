from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import PIL.Image as Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from model import Net
import pandas as pd
from torchvision import datasets, transforms
from data import data_transforms



state_dict = torch.load('/workspace/gtsrb_pytorch/model/model_40.pth')
model = Net()
model.load_state_dict(state_dict)
model.eval()


if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
	use_gpu = False
	print("Using CPU")


val_data_path = '/workspace/dabs/data/adv_data/traffic_sign/07_01_2023/traffic_budget_budget=0.005/val'
# val_data_path = '/workspace/gtsrb_pytorch/data/val_images'
# Apply data transformations on the training images to augment dataset
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_data_path,transform=data_transforms,is_valid_file=lambda s: 'view' in s),
    batch_size=32, shuffle=False, num_workers=4, pin_memory=use_gpu)
   
if use_gpu:
    model.cuda()

model.eval()
validation_loss = 0
correct = 0
for data, target in tqdm(val_loader):
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        output = model.forward_original(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

print('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format( correct, len(val_loader.dataset),100. * correct / len(val_loader.dataset)))
