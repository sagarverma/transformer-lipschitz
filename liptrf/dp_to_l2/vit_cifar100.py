import os 
import argparse 
import pickle as pkl 
import numpy as np 
import csv 

import timm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from liptrf.models.vit import L2Attention
from liptrf.models.layers import LinearX
from liptrf.models.timm_vit import VisionTransformer as ViT


def train(args, model, device, train_loader,
          optimizer, epoch, criterion, finetune=False):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()

        torch.cuda.empty_cache()

    train_loss /= len(train_loader.dataset)
    train_samples = len(train_loader.dataset)

    print(f"Epoch: {epoch}, Train set: Average loss: {train_loss:.4f}, " +
          f"Accuracy: {correct}/{train_samples} " +
          f"({100.*correct/train_samples:.0f}%), " +
          f"Error: {(train_samples-correct)/train_samples * 100:.2f}%")


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    test_samples = len(test_loader.dataset)
    lip = model.lipschitz().item()

    print(f"Test set: Average loss: {test_loss:.4f}, " +
          f"Accuracy: {correct}/{test_samples} " +
          f"({100.*correct/test_samples:.0f}%), " +
          f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
          f"Lipschitz {lip:4f} \n")
    return 100.*correct/test_samples, test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 ViT')
    parser.add_argument('--task', type=str, default='train',
                        help='train/retrain/extract/test')

    parser.add_argument('--layers', type=int, default=12)
    
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--opt', type=str, default='adam',
                        help='adam/sgd')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cores to use')
    parser.add_argument('--cos', action='store_false', 
                        help='Train with cosine annealing scheduling')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to use')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='data path of CIFAR100')
    parser.add_argument('--dp_weight_path', type=str, required=True,
                        help='weight path of ViT trained with DP attention')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.gpu)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = datasets.CIFAR100(
        root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = datasets.CIFAR100(
        root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model =  ViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, lmbda=1, num_classes=100)
    args.layers = 12
    weight = torch.load(args.dp_weight_path)
    model.load_state_dict(weight, strict=False)
    criterion = nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = False 

    for i in range(args.layers):
        model.blocks[i].attn = L2Attention(dim=192, heads=3, dropout=0.1)

    model = model.to(device)

    if args.opt == 'adam': 
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), weight_decay=5e-5)
    elif args.opt == 'sgd': 
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=0.9,
                        weight_decay=0.0) 
    # use cosine or reduce LR on Plateau scheduling
    if not args.cos:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                        patience=3, verbose=True, 
                                                        min_lr=1e-3*1e-5, factor=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, 
                                                         eta_min=1e-5)
    if args.task == 'train':
        weight_path = os.path.join(args.dp_weight_path.replace('.pt', '_L2-Adapted.pt'))
        fout = open(weight_path.replace('.pt', '.csv').replace('weights', 'logs'), 'w')
        w = csv.writer(fout)

        best_acc = -1
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader,
                  optimizer, epoch, criterion, False)
            acc, loss = test(args, model, device, test_loader, criterion)
            w.writerow([epoch, acc, loss, 'NaN'])
        
            if args.cos:
                scheduler.step(epoch-1)
            else:
                scheduler.step(loss)
        
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), weight_path)

    if args.task == 'test':
        weight = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(weight)
        test(args, model, device, test_loader, criterion)
            

if __name__ == '__main__':
    main()