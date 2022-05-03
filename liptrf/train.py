import time
from liptrf import lipschitz 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision

from liptrf.models.vit import ViT


torch.manual_seed(42)

DOWNLOAD_PATH = './data/mnist'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000

transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)


def train_epoch(model, criterion, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"[{i * len(data)} / {total_samples} ({100 * i / len(data_loader)} )]  Loss: {loss.item()}")
            loss_history.append(loss.item())

def train_robust(model, criterion, optimizer, data_loader, loss_history, epsilon):
    total_samples = len(data_loader.dataset)
    model.eval()

    # Calculate Lipshitz of the network
    lipschitz = model.lipschitz()

    model.train()
    for i, (data, target) in enumerate(data_loader):
        data = data.cuda() + epsilon
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        output += 2 ** 0.5 * epsilon * lipschitz
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"[{i * len(data)} / {total_samples} ({100 * i / len(data_loader)} )]  Loss: {loss.item()}")        
            loss_history.append(loss.item())

    print(f"Lipschitz: {lipschitz}")

def evaluate(model, criterion, data_loader, loss_history):
    model.eval()

    # Calculate Lipshitz of the network
    lipschitz = model.lipschitz()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print(f"\nAverage test loss: {avg_loss}  Accuracy: {correct_samples} /{total_samples} ({100.0 * correct_samples / total_samples} )\n")
    print(f"Lipschitz: {lipschitz}")

N_EPOCHS = 25
EPSILON = 1.58 
WARMUP = 1 

start_time = time.time()
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128, attention_type='L2').cuda()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    if epoch < WARMUP:
        print('Epoch:', epoch)
        train_epoch(model, criterion, optimizer, train_loader, train_loss_history)
        evaluate(model, criterion, test_loader, test_loss_history)
    else:
        print ('Robust Epoch:', epoch)
        train_robust(model, criterion, optimizer, train_loader, train_loss_history, EPSILON)
        evaluate(model, criterion, test_loader, test_loss_history)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')