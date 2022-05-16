import os 
import random 
import argparse

import torch 
import timm
import torch.nn as nn 
import torch.optim as optim
import torch.distributed as dist 


from liptrf.models.vit import L2Attention
from liptrf.utils.imagenet_dataloader import get_dataloaders 


attention_io = {}
def get_activation(name):
    def hook(model, input, output):
        attention_io[name] = {"in": input[0].detach(),
                              "out": output.detach()}
    return hook 

def adapt(args, dp_model, l2_model, student_l2_models, device, train_loader,
          student_l2_optims, epoch, criterion, finetune=False):
    student_l2_models = [student_l2_model.train() for student_l2_model in student_l2_models]
    dp_model.eval()

    samples = 0
    student_l2_train_loss = [0] * len(student_l2_models)
    for batch_idx, (data, target) in enumerate(train_loader):
        with torch.no_grad():
            data = data.to(device)
            output = dp_model(data)
            samples += data.shape[0]
            del data, target, output

        [student_l2_optim.zero_grad() for student_l2_optim in student_l2_optims]
        for i in range(12):
            teacher_l2_data = attention_io[f"a{i}"]["in"]
            teacher_l2_target = attention_io[f"a{i}"]["out"]
            student_l2_output = student_l2_models[i](teacher_l2_data)
            student_l2_loss = criterion(student_l2_output, teacher_l2_target)
            student_l2_loss.backward()
            student_l2_train_loss[i] += student_l2_loss.item()
            student_l2_optims[i].step()

    for i in range(12):
        l2_model.blocks[i].attn = student_l2_models[i]

    student_l2_train_loss = [student_l2_loss / samples for student_l2_loss in student_l2_train_loss]

    print(f"Epoch: {epoch}, Train set: Average losses")
    for i in range(12):
        print(f"Attention {i}: {student_l2_train_loss[i]}")


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    test_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) - 1
            test_samples += data.shape[0]
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log_probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            torch.cuda.empty_cache()

    test_loss /= test_samples

    print(f"Test set: Average loss: {test_loss:.4f}, " +
          f"Accuracy: {correct}/{test_samples} " +
          f"({100.*correct/test_samples:.0f}%), " +
          f"Error: {(test_samples-correct)/test_samples * 100:.2f}% " +
          f"Lipschitz {model.lipschitz().item():4f} \n")
    return 100.*correct/test_samples


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST ViT')
    parser.add_argument('--task', type=str, default='adapt',
                        help='adapt/test')
    
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

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to use')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--data_path', type=str, required=True,
                        help='data path of MNIST')
    parser.add_argument('--weight_path', type=str, required=True, 
                        help='weight path save directory')
    parser.add_argument('--train_samples', type=int, default=1878)
    parser.add_argument('--val_samples', type=int, default=48)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.gpu)
    gpus = torch.cuda.device_count()

    train_loader, test_loader = get_dataloaders(args)
    

    l2_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    dp_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    dp_model.eval()

    student_l2_criterion = nn.MSELoss()
    student_l2_optims = []
    student_l2_schedulers = []
    student_l2_models = []
    for i in range(12):
        dp_model.blocks[i].attn.register_forward_hook(get_activation(f'a{i}'))
        student_l2_model = L2Attention(dim=768, heads=12).to(i % (gpus-1) + 1)

        if args.opt == 'adam': 
            student_l2_optimizer = optim.Adam(student_l2_model.parameters(), lr=args.lr)
        elif args.opt == 'sgd': 
            student_l2_optimizer = optim.SGD(student_l2_model.parameters(), lr=args.lr, 
                            momentum=0.9,
                            weight_decay=0.0) 
        student_l2_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_l2_optimizer,
                                                        milestones=[50, 60,
                                                                    70, 80],
                                                        gamma=0.2)
        student_l2_models.append(student_l2_model)
        student_l2_optims.append(student_l2_optimizer)
        student_l2_schedulers.append(student_l2_scheduler)

    criterion = nn.CrossEntropyLoss()

    if args.task == 'adapt':
        weight_path = os.path.join(args.weight_path, f"vit_base_patch16_224_att-L2_adaptedFrom_DP.pt")

        if not os.path.exists(args.weight_path):
            os.mkdir(args.weight_path)

        best_acc = -1
        for epoch in range(1, args.epochs + 1):
            adapt(args, dp_model, l2_model, student_l2_models, device, train_loader,
                  student_l2_optims, epoch, student_l2_criterion, False)
            acc = test(args, l2_model, device, test_loader, criterion)
            
            for student_l2_scheduler in student_l2_schedulers:
                student_l2_scheduler.step()

            if acc > best_acc:
                torch.save(l2_model.state_dict(), weight_path)
            
    if args.task == 'test':
        weight = torch.load(args.weight_path)
        l2_model.load_state_dict(weight)
        test(args, l2_model, device, test_loader, criterion)

if __name__ == '__main__':
    main()