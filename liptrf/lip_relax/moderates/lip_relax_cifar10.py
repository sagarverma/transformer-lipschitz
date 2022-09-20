import os 
import argparse 
import pickle as pkl 
import numpy as np
import tqdm

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from liptrf.models.moderate import CIFAR10_C6F2_ReLU
from liptrf.models.layers.linear import LinearX
from liptrf.models.layers.conv import Conv2dX

from liptrf.utils.evaluate import evaluate_pgd
from liptrf.utils.local_bound import evaluate


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
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
    
        output = model(data)
    
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    
        torch.cuda.empty_cache()

    test_samples = len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)
    test_samples = len(test_loader.dataset)
    lip = model.lipschitz().item()

    print(f"Test set: Average loss: {test_loss:.4f}, " +
          f"Accuracy: {correct}/{test_samples} " + 
          f"({100.*correct/test_samples:.0f}%), " +
          f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
          f"Lipschitz {lip:4f} \n")
    return 100.*correct/test_samples


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Moderate')

    parser.add_argument('--power_iter', type=int, default=5)
    parser.add_argument('--lc_gamma', default=0.1, type=float)
    parser.add_argument('--lc_alpha', default=0.01, type=float)
    parser.add_argument('--eta', default=1e-2, type=float)
    parser.add_argument('--lr', default=1.2, type=float)
    parser.add_argument('--lipr_epochs', default=2, type=int)
    parser.add_argument('--proj_epochs', default=100, type=int)

    parser.add_argument('--data', default='cifar10', type=str)
    parser.add_argument('--model', default='standrelu', type=str)
    
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cores to use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to use')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='data path of MNIST')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='weight path of trained network')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.gpu)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model =  CIFAR10_C6F2_ReLU(power_iter=args.power_iter, lmbda=1, lc_gamma=args.lc_gamma, lc_alpha=args.lc_alpha, lr=args.lr, eta=args.eta)
    weight = torch.load(args.weight_path)
    model.load_state_dict(weight, strict=False)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    

    # test(args, model, pgd_attack, device, test_loader, criterion)
    layers = []
    for _ in range(args.lipr_epochs):
        for layer in model.modules():
            if isinstance(layer, LinearX) or isinstance(layer, Conv2dX):
                layers.append(layer)

    

    for layer in layers:
        layer.weight_t = layer.weight.clone().detach()
        # wt = torch.abs(layer.weight_t)
        # wt[wt > layer.lc_alpha * torch.linalg.norm(wt)] += layer.lc_gamma
        # layer.weight_t = wt * torch.sign(layer.weight_t)

        # layer.weight_t = (torch.abs(layer.weight_t) + layer.lc_gamma) * torch.sign(layer.weight_t)

    for lipr_epoch in range(args.lipr_epochs):
        [layer.prox() for layer in layers]
        for layer in layers:
            layer.proj_done = False

        for proj_epoch in tqdm.tqdm(range(args.proj_epochs)):
            for layer in layers:
                layer.proj_weight_old = layer.proj_weight.clone().detach()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, _ = data.to(device), target.to(device)
                _ = model(data)
                for layer in layers:
                    if not layer.prox_done and not layer.proj_done:
                        layer.proj()
                # break

            for layer in layers:
                if torch.linalg.norm(layer.proj_weight - layer.proj_weight_old) < 1e-7 * torch.linalg.norm(layer.proj_weight):
                    layer.proj_done = True
        
        for layer in layers:
            if not layer.prox_done:
                layer.update()

        for layer in layers:
            if torch.linalg.norm(layer.weight_t - layer.weight_old) < 1e-4 * torch.norm(layer.weight_t):
                layer.prox_done = True
        
    for layer in layers:
        if not layer.prox_done:
            layer.weight = nn.Parameter(layer.prox_weight.clone().detach())

    test(args, model, device, test_loader, criterion)
    evaluate(test_loader, model, 36/255, 10, args, None, u_test=None, save_u=False)  
    evaluate_pgd(test_loader, model, epsilon=36/255., niter=10, alpha=36/255/4)

    weight_path = args.weight_path.replace('.pt', f"_lc_alpha-{args.lc_alpha}_eta-{args.eta}_lc_gamma-{args.lc_gamma}_lr-{args.lr}.pt")    
    torch.save(model.state_dict(), weight_path)

    
    

if __name__ == '__main__':
    main()