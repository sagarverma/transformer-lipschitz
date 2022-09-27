import os 
import argparse 
import pickle as pkl 
import numpy as np
import csv

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from liptrf.models.vit import ViT

from liptrf.utils.evaluate import evaluate_pgd

def train(args, model, device, train_loader,
          optimizer, epoch, criterion):
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
        worst_logit = output + 2**0.5 * 1.58 * lip * (1 - one_hot)
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
    parser = argparse.ArgumentParser(description='PyTorch MNIST ViT')
    parser.add_argument('--task', type=str, default='train',
                        help='train/retrain/extract/test')

    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--lmbda', type=float, default=1.)
    parser.add_argument('--power_iter', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--attention_type', type=str, default='L2',
                        help='L2/DP')

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
    parser.add_argument('--model', type=str, default='vit')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to use')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='data path of MNIST')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='weight path of MNIST')

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
                dim=128, depth=args.layers, heads=8, mlp_ratio=4, 
                attention_type=args.attention_type, 
                power_iter=args.power_iter, lmbda=args.lmbda, 
                device=device).to(device)
    print (sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    if args.opt == 'adam': 
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=0.9,
                        weight_decay=0.0) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50, 60,
                                                                 70, 80],
                                                     gamma=0.2)

    if args.task == 'train':
        if not args.relax:
            weight_path = os.path.join(args.weight_path, f"MNIST_{args.model}_seed-{args.seed}_layers-{args.layers}")
        else:
            weight_path = os.path.join(args.weight_path, f"MNIST_{args.model}_seed-{args.seed}_layers-{args.layers}_relax-{args.lmbda}_warmup-{args.warmup}")
        weight_path += f"_att-{args.attention_type}.pt"
        
        fout = open(weight_path.replace('.pt', '.csv').replace('weights', 'logs'), 'w')
        w = csv.writer(fout)

        if not os.path.exists(args.weight_path):
            os.mkdir(args.weight_path)

        best_acc = -1
        best_state = None
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader,
                  optimizer, epoch, criterion)
            acc, loss, lip = test(args, model, device, test_loader, criterion)
            w.writerow([epoch, acc, loss, lip])
            scheduler.step()
            if acc > best_acc and epoch >= args.warmup:
                best_acc = acc
                best_state = model.state_dict()
                torch.save(model.state_dict(), weight_path)
        
        fout.close() 
        model.load_state_dict(best_state)
        model.eval()
        test(args, model, device, test_loader, criterion)
        evaluate_pgd(test_loader, model, epsilon=1.58, niter=100, alpha=1.58/4, device=device)

    if args.task == 'test':
        weight = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(weight)
        model.eval()
        test(args, model, device, test_loader, criterion)
        evaluate_pgd(test_loader, model, epsilon=1.58, niter=100, alpha=1.58/4, device=device)

if __name__ == '__main__':
    main()