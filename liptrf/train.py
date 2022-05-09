import time
from liptrf import lipschitz 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision

from liptrf.models.vit import ViT
from liptrf.models.linear_toy import Net as LinearNet
from liptrf.models.conv_toy import Net as ConvNet

from liptrf.utils.evaluate import evaluate_pgd
from liptrf.utils.training import lr_exp

# TODO: Arguments YAML config 
# TODO: Use args not hard code
torch.manual_seed(42)

DOWNLOAD_PATH = './data/mnist'
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 2048
EPOCHS = 300
RAMPUP = 150
WARMUP = 10
OPT = "adam"
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
LR = 0.001 
END_LR = 5e-6
LR_SCHEDULER = 'exp'
STEP_SIZE = 10 
GAMMA = 0.5
WD_LIST = []


EPSILON = 1.58
EPSILON_TRAIN = 1.75
STARTING_EPSILON = 0
SCHEDULE_LENGTH = RAMPUP
KAPPA = 0.0
STARTING_KAPPA = 1.0
KAPPA_SCEDULER_LENGTH = RAMPUP
NITER = 100 
ALPHA = EPSILON / 4

DEPTH = 6
HEADS = 8

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

def train_robust(model, criterion, optimizer, data_loader, loss_history, epsilon, kappa):
    total_samples = len(data_loader.dataset)
    standard_losses = []
    robust_losses = []
    lipschitzs = []

    model.train()
    for i, (data, target) in enumerate(data_loader):
        start_epsilon = epsilon + i / len(data_loader) * (EPSILON_TRAIN - STARTING_EPSILON )/ SCHEDULE_LENGTH
        # start_kappa = kappa + i/ len(data_loader)*  (KAPPA - STARTING_KAPPA) / SCHEDULE_LENGTH

        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        
        lipschitz = model.lipschitz().item()
        # print (lipschitz)
        kW = lipschitz * model.mlp_head[1].weight.clone().detach().T
        j = torch.argmax(output, dim=1)
        y_j = torch.max(output, dim=1, keepdim=True)[0]
        kW_j = kW.T[j]
        kW_ij = kW_j[:,:,None] - kW[None]
        
        K_ij = torch.sqrt(torch.sum(kW_ij * kW_ij, dim=1))
        # # lipschitz_constant = torch.where(torch.eq(output, y_j), torch.zeros_like(K_ij) - 1., K_ij)
        
        y_bot_i = output + start_epsilon * K_ij
        y_bot_i = torch.where(torch.eq(output, y_j), -np.infty + torch.zeros_like(y_bot_i), y_bot_i)
        y_bot = torch.max(y_bot_i, dim=1, keepdims=True)[0]

        y_pred = torch.cat([output, y_bot], dim=1)
        standard_loss = criterion(y_pred[:, :-1], target)

        y_pred_soft = torch.softmax(y_pred, dim=1)
        new_ground_truth = torch.cat([torch.softmax(y_pred[:, :-1], dim=1), 
                                      torch.zeros(output.shape[0], 1).cuda()], dim=1)
        robust_loss = F.kl_div(y_pred_soft, new_ground_truth)
        # onehot = one_hot(target)
        # output[onehot == 0] += (2**0.5) * start_epsilon * lipschitz
        loss = standard_loss + 1.5 * robust_loss
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        standard_losses.append(standard_loss.item())
        robust_losses.append(robust_loss.item())
        lipschitzs.append(lipschitz)
        
    print(f"Avg. Standard Loss: {np.mean(standard_losses):.2f} Avg. Robust Loss: {np.mean(robust_losses):.2f} Avg. Lipschitz {np.mean(lipschitzs):.2f}")        
   

def evaluate(model, criterion, data_loader, loss_history, epsilon):
    model.eval()

    # Calculate Lipshitz of the network
    lipschitz = model.lipschitz()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    correct_samples_eps = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

            output_eps = model(data + epsilon)
            _, pred_eps = torch.max(output_eps, dim=1)
            correct_samples_eps += pred_eps.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print(f"\nAverage test loss: {avg_loss:.2f}  \
 Accuracy: {correct_samples} /{total_samples} ({(100.0 * correct_samples / total_samples):.2f} )\
 Lipschitz {lipschitz:.2f} \
 Certified Accuracy {(100.0 * correct_samples_eps / total_samples):.2f} \n")


eps_schedule = np.linspace(STARTING_EPSILON,
                            EPSILON_TRAIN,                                
                            SCHEDULE_LENGTH)
kappa_schedule = np.linspace(STARTING_KAPPA, 
                             KAPPA,                                
                             KAPPA_SCEDULER_LENGTH)

start_time = time.time()
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=128, depth=DEPTH, heads=HEADS, mlp_ratio=4, attention_type='L2').cuda()
# model = LinearNet().cuda()
# model = ConvNet().cuda()
if OPT == 'adam': 
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif OPT == 'sgd': 
    optimizer = optim.SGD(model.parameters(), lr=LR, 
                    momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY) 
print(optimizer)
if LR_SCHEDULER == 'step':
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
elif LR_SCHEDULER =='multistep':
    lr_scheduler = MultiStepLR(optimizer, milestones=WD_LIST, gamma=GAMMA)
elif (LR_SCHEDULER == 'exp'):
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_exp(LR, END_LR, step, EPOCHS))  
print(lr_scheduler)

criterion = nn.CrossEntropyLoss()


train_loss_history, test_loss_history = [], []
for epoch in range(1, EPOCHS + 1):
    if epoch < WARMUP:
        epsilon = 0
        epsilon_next = 0
    elif WARMUP <= epoch < WARMUP + len(eps_schedule) and STARTING_EPSILON is not None: 
        epsilon = float(eps_schedule[epoch - WARMUP])
        epsilon_next = float(eps_schedule[np.min((epoch + 1 - WARMUP, len(eps_schedule)-1))])
    else:
        epsilon = EPSILON_TRAIN
        epsilon_next = EPSILON_TRAIN

    if epoch < WARMUP:
        kappa = 1
        kappa_next = 1
    elif WARMUP <= epoch < WARMUP + len(kappa_schedule):
        kappa = float(kappa_schedule[epoch - WARMUP])
        kappa_next = float(kappa_schedule[np.min((epoch + 1 - WARMUP, len(kappa_schedule)-1))])
    else:
        kappa = KAPPA
        kappa_next = KAPPA

    if epoch < WARMUP:
        print('Epoch:', epoch)
        train_epoch(model, criterion, optimizer, train_loader, train_loss_history)
        evaluate(model, criterion, test_loader, test_loss_history, EPSILON)
    else:
        print (f"Robust Epoch: {epoch} Epsilon: {epsilon}")
        train_robust(model, criterion, optimizer, train_loader, train_loss_history, epsilon, kappa)
        evaluate(model, criterion, test_loader, test_loss_history, EPSILON)
        evaluate_pgd(test_loader, model, EPSILON, NITER, ALPHA)
        print ("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    if LR_SCHEDULER == 'step': 
        if max(epoch - (RAMPUP + WARMUP - 1) + 1, 0):
            print("LR DECAY STEP")
        lr_scheduler.step(epoch=max(epoch - (RAMPUP + WARMUP - 1) + 1, 0))
    elif LR_SCHEDULER =='multistep' or LR_SCHEDULER =='exp':
        print("LR DECAY STEP")
        lr_scheduler.step()      
    else:
        raise ValueError("Wrong LR scheduler")

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')