import os 
import argparse 
import pickle as pkl 
import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from models.vit import ViT, L2Attention
from models.linear_toy import LinearX


def liprex(args, model, layers, device, train_loader, criterion):
    [layer.prox() for layer in layers]

    with torch.no_grad():
        for proj_epoch in range(args.proj_epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                [layer.proj() for layer in layers]

    [layer.update() for layer in layers]

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

    print(f"Test set: Average loss: {test_loss:.4f}, " +
          f"Accuracy: {correct}/{test_samples} " +
          f"({100.*correct/test_samples:.0f}%), " +
          f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
          f"Lipschitz {model.lipschitz().item():4f} \n")
    return 100.*correct/test_samples

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST ViT')

    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--relax', default=1., type=float)
    parser.add_argument('--eta', default=1e-7, type=float)
    parser.add_argument('--lr', default=1., type=float)
    parser.add_argument('--lipr_epochs', default=10, type=int)
    parser.add_argument('--proj_epochs', default=10, type=int)
    
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
    parser.add_argument('--l2_weight_path', type=str, required=True,
                        help='weight path of ViT trained/adapted with/to L2 attention')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.gpu)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(args.data_path, train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, 
                                                num_workers=args.num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size, 
                                                num_workers=args.num_workers, shuffle=False)

    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=128, depth=args.layers, heads=8, mlp_ratio=4, attention_type='L2', lmbda=1).to(device)
    layers =[]
    for layer in model.modules():
        if isinstance(layer, LinearX):
            layer.relax = args.relax
            layer.eta = args.eta 
            layer.lr = args.lr 
            layers.append(layer)
    weight = torch.load(args.l2_weight_path)
    model.load_state_dict(weight)
    model.eval()

    criterion = nn.CrossEntropyLoss()


    weight_path = args.l2_weight_path.replace('.pt', f"_relax-{args.relax}_eta-{args.eta}_lr-{args.lr}.pt")

    best_acc = -1
    for lipr_epoch in range(1, args.lipr_epochs + 1):
        with torch.no_grad():
            liprex(args, model, layers, device, train_loader, criterion)
            acc = test(args, model, device, test_loader, criterion)
        
        if acc > best_acc:
            torch.save(model.state_dict(), weight_path)

if __name__ == '__main__':
    main()