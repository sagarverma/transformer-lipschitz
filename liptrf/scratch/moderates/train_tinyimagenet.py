import os
import copy
import argparse 
import pickle as pkl 
import numpy as np
import csv

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms

from liptrf.models.moderate import TinyImageNet_8C2F_ReLUx

from liptrf.utils.evaluate import evaluate_pgd


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

        with torch.no_grad():
            if args.relax and epoch > args.warmup:
                model.lipschitz()
                model.apply_spec()
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
    
    lip = model.lipschitz()
    verified = 0

    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        # print (output.max())
        one_hot = F.one_hot(target, num_classes=output.shape[-1])
        worst_logit = output + 2**0.5 * 36/255 * lip * (1 - one_hot)
        worst_pred = worst_logit.argmax(dim=1, keepdim=True)
        verified += worst_pred.eq(target.view_as(worst_pred)).sum().item()

        torch.cuda.empty_cache()

    test_samples = len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)
    test_samples = len(test_loader.dataset)
    
    print(f"Test set: Average loss: {test_loss:.4f}, " +
          f"Accuracy: {correct}/{test_samples} " + 
          f"({100.*correct/test_samples:.2f}%), " +
          f"Verified: {100.*verified/test_samples:.2f}%, " +
          f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
          f"Lipschitz {lip:4f}")
    
    return 100.*correct/test_samples, 100.*verified/test_samples, lip



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch TinyImageNet')
    parser.add_argument('--task', type=str, default='train',
                        help='train/retrain/extract/test')

    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--lmbda', type=float, default=1.)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--power_iter', type=int, default=10)

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
    parser.add_argument('--model', type=str, default='8c2f_relux')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to use')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='data path of TinyImageNet')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='weight path of TinyImageNet')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.gpu)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
    trainset = datasets.ImageFolder(root=f'{args.data_path}/tiny-imagenet-200/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
    testset = datasets.ImageFolder(root=f'{args.data_path}/tiny-imagenet-200/val', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.model == '8c2f_relux':
        model = TinyImageNet_8C2F_ReLUx(lmbda=args.lmbda, power_iter=args.power_iter).to(device)

    criterion = nn.CrossEntropyLoss()
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
        if not args.relax:
            weight_path = os.path.join(args.weight_path, f"TinyImageNet_{args.model}_seed-{args.seed}")
        else:
            weight_path = os.path.join(args.weight_path, f"TinyImageNet__{args.model}_seed-{args.seed}_relax-{args.lmbda}_warmup-{args.warmup}")
        weight_path += f".pt"

        fout = open(weight_path.replace('.pt', '.csv').replace('weights', 'logs'), 'w')
        w = csv.writer(fout)

        if not os.path.exists(args.weight_path):
            os.mkdir(args.weight_path)

        best_acc = -1
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader,
                  optimizer, epoch, criterion, False)
            acc, loss, lip = test(args, model, device, test_loader, criterion)
            w.writerow([epoch, acc, loss, lip])
        
            if args.cos:
                scheduler.step(epoch-1)
            else:
                scheduler.step(loss)
        
            if acc > best_acc and epoch >= args.warmup:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weight_path)
        
        fout.close()
        model.load_state_dict(best_state)
        model.eval()
        test(args, model, device, test_loader, criterion)
        evaluate_pgd(test_loader, model, epsilon=36/255, niter=100, alpha=36/255/4, device=device)

    if args.task == 'test':
        weight = torch.load(args.weight_path, map_location=device)
        if 'state_dict' in weight.keys():
            layers = list(model.children())
            for k in weight['state_dict'].keys():
                # print (k)
                idx, name = k.split('.')
                layers[int(idx)].__dict__[name] = weight['state_dict'][k]
        else:
            model.load_state_dict(weight)
        model.eval()
        test(args, model, device, test_loader, criterion)
        evaluate_pgd(test_loader, model, epsilon=36/255, niter=100, alpha=36/255/4, device=device)

if __name__ == '__main__':
    main()