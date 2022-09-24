import torch
import timm 
from liptrf.utils.imagenet_dataloader import get_dataloaders 


attention_io = {}
def get_activation(name):
    def hook(model, input, output):
        attention_io[name] = {"in": input[0].detach(),
                              "out": output.detach()}
    return hook 
    
def extract(args, model, device, train_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    test_samples = 0
    with torch.no_grad():
        for data, target in train_loader:
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