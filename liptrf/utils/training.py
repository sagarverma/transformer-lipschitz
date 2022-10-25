def lr_exp(start_lr, end_lr, epoch, max_epoch, more=25):
    if epoch >= (max_epoch//2 + more):
        scalar = (end_lr/start_lr)**((float(epoch)-(max_epoch//2 +more-1))/(max_epoch//2 - more))
    else:
        scalar = 1.0
    return scalar  