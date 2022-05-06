from math import ceil
import numpy as np
import torch
import torch.nn as nn 

from advertorch.attacks import L2PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval


# TODO: use args not hard code 

def evaluate_pgd(loader, model, epsilon, niter, alpha):
    model.eval()
    accs = []

    adversary = L2PGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, 
        nb_iter=niter, eps_iter=alpha, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        with ctx_noparamgrad_and_eval(model):
            X_pgd = adversary.perturb(X, y)
            
        out = model(X_pgd)
        acc = (out.data.max(1)[1] == y).float().sum() / X.size(0)
        accs.append(acc.data.cpu().item())
    print(f'PGD Accuracy {np.mean(accs) * 100:.2f}')