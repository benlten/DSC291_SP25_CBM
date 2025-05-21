import time
import os
import sys
import argparse
sys.path.append('.')
# from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

from utils.datasets import CelebAHQ_dataset

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training concept classifier for evaluation')
parser.add_argument("-c", "--classes", action='append', help='names of classes from CelebAHQ to use for training')
parser.add_argument("-b", "--batch-size", default=512, type=int, help='batch size for training')
parser.add_argument("-n", "--num-epochs", default=5, type=int, help='number of epochs for training')
parser.add_argument("-d", "--dataset", default='celebahq', type=str, help='which dataset to use - celebahq or celeba64')
parser.add_argument("-m", "--model-type", default='rn18', type=str, help='which model to use for training')
parser.add_argument("--base-root", default='/expanse/lustre/projects/ddp390/akulkarni', type=str, help='path to datasets folder')
parser.add_argument("--load-pretrained", action='store_true', default=False, help='try to load already trained model as initialization')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
print(f'using device {device}')


# base_root = '/expanse/lustre/projects/ddp390/akulkarni'
if args.dataset == 'celebahq':
    size = 256
    if args.model_type == 'vit_l_16':
        size = 512
    base_root = args.base_root
    img_root = base_root+'/datasets/CelebAMask-HQ/CelebA-HQ-img'
    train_file = base_root+'/datasets/CelebAMask-HQ/train.txt'
    test_file = base_root+'/datasets/CelebAMask-HQ/test.txt'
    transforms_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

elif args.dataset == 'celeba64':
    size = 64
    if args.model_type == 'vit_l_16':
        size = 512
    base_root = args.base_root
    img_root = base_root+'/datasets/img_align_celeba'
    train_file = base_root+'/datasets/celeba64_train_annotations.txt'
    test_file = base_root+'/datasets/celeba64_val_annotations.txt'

    transforms_train = transforms.Compose([
        transforms.Resize((size)),
        transforms.CenterCrop((size, size)),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((size)),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = CelebAHQ_dataset(img_root, train_file, set_of_classes=args.classes, transform=transforms_train)
test_dataset = CelebAHQ_dataset(img_root, test_file, set_of_classes=args.classes, transform=transforms_test)

test_batch_size = 1000
if args.model_type == 'vit_l_16':
    test_batch_size = args.batch_size
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

if len(args.classes) == 1:
    save_name = args.classes[0]
    args.classes = [f'not {args.classes[0]}', f'{args.classes[0]}']
else:
    save_name = args.classes[0].split('_')[-1]

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

print('Class names:', args.classes)
print(f'using save name: {save_name}')

if args.model_type == 'rn18':
    model = models.resnet18(weights='DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(args.classes)) # binary classification (num_of_class == 2)
elif args.model_type == 'vit_l_16':
    ## getting one of the best imagenet models (as of 11th october) from https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
    model = models.vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1')
    num_features = model.heads.head.in_features
    model.heads = nn.Linear(num_features, len(args.classes))
model = model.to(device)

if args.load_pretrained:
    save_path_ = f'models/checkpoints/{args.dataset}_{save_name}_{args.model_type}_conclsf.pth'
    if os.path.exists(save_path_):
        print(f'loading pretrained checkpoint from {save_path_}')
        model.load_state_dict(torch.load(save_path_), strict=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs = args.num_epochs
start_time = time.time()
best_acc = 0.0

for epoch in range(num_epochs):
    """ Training Phase """
    model.train()

    running_loss = 0.
    running_corrects = 0

    # load a batch data of images
    # for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
    for i, (inputs, labels) in enumerate((train_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # get loss value and update the network weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    """ Test Phase """
    model.eval()

    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        # for _, (inputs, labels) in enumerate(tqdm(test_dataloader)):
        for _, (inputs, labels) in enumerate((test_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_path = f'models/checkpoints/{args.dataset}_{save_name}_{args.model_type}_conclsf.pth'
            print(f'saving model with test accuracy {best_acc}')
            torch.save(model.state_dict(), save_path)
