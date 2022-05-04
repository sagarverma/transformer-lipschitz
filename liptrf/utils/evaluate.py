import numpy as np
import torch.nn as nn 

from advertorch.attacks import L2PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval


def evaluate_pgd(loader, model):
    model.eval()
    errors = []

    adversary = L2PGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.58/4, 
        nb_iter=100, eps_iter=5., rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        with ctx_noparamgrad_and_eval(model):
            X_pgd = adversary.perturb(X, y)
            
        out = model(X_pgd)
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)
        errors.append(err.data.cpu().item())
    print(f' * Error {np.mean(errors)}')
    return np.mean(errors)
