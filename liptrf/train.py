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
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 2048
N_EPOCHS = 300
EPSILON = 0
WARMUP = 0
CLAMP = 1
DEPTH = 6

def one_hot(batch, depth=10):
    ones = torch.eye(depth, device=batch.device)
    return ones.index_select(0,batch)


transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)


def train_epoch(model, criterion, optimizer, data_loader, loss_history, clamp):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(-1* clamp, clamp)

        if i % 100 == 0:
            print(f"[{i * len(data)} / {total_samples} ({100 * i / len(data_loader)} )]  Loss: {loss.item()}")
            loss_history.append(loss.item())

def train_robust(model, criterion, optimizer, data_loader, loss_history, epsilon, clamp):
    total_samples = len(data_loader.dataset)
    model.eval()

    # Calculate Lipshitz of the network
    lipschitz = model.lipschitz().item() 

    model.train()
    for i, (data, target) in enumerate(data_loader):
        epsilon += 0.0001
        data = data.cuda() + epsilon
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        onehot = one_hot(target)
        output[onehot == 0] += (2 ** 0.5) * epsilon * lipschitz
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(-1* clamp, clamp)

        if i % 100 == 0:
            print(f"[{i * len(data)} / {total_samples} ({100 * i / len(data_loader)} )]  Loss: {loss.item()}")        
            loss_history.append(loss.item())

    print(f"Lipschitz: {lipschitz}")

def evaluate(model, criterion, data_loader, loss_history):
    model.eval()

    # Calculate Lipshitz of the network
    lipschitz = model.lipschitz().item()
    
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



start_time = time.time()
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=32, depth=DEPTH, heads=1, mlp_ratio=2, attention_type='L2').cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for param in model.parameters():
        param.clamp_(-1* CLAMP, CLAMP)

train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    if epoch < WARMUP:
        print('Epoch:', epoch)
        train_epoch(model, criterion, optimizer, train_loader, train_loss_history, CLAMP)
        evaluate(model, criterion, test_loader, test_loss_history)
    else:
        print ('Robust Epoch:', epoch)
        train_robust(model, criterion, optimizer, train_loader, train_loss_history, EPSILON, CLAMP)
        evaluate(model, criterion, test_loader, test_loss_history)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')