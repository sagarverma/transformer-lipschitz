import os
import io
 
from PIL import Image
import webdataset as wds
from torchvision import datasets, transforms


class Byte2Image(object):
    def __call__(self, sample):
        return Image.open(io.BytesIO(sample))

def identity(x):
    return x


def get_dataloaders(args):
    train_transform = transforms.Compose(
                [   Byte2Image(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    val_transform = transforms.Compose(
                [   Byte2Image(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    
    train_samples = args.train_samples.zfill(4)
    train_dataset = (wds.WebDataset(os.path.join(args.data_path, 'train/train-{0000..' + train_samples + '}.tar'))
            .shuffle(True)
            .decode("pil")
            .to_tuple("x.img.pil y.cls")
            .map_tuple(train_transform, identity)
            .batched(args.batch_size, partial=False))
    
    val_samples = args.val_samples.zill(2)
    val_dataset = (
            wds.WebDataset(os.path.join(args.data_path, 'val/val-{00..' + val_samples + '}.tar'))
            .shuffle(False)
            .decode("pil")
            .to_tuple("x.img.pil y.cls")
            .map_tuple(val_transform, identity)
            .batched(args.batch_size, partial=False)
        )
    train_loader =  wds.WebLoader(
            train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.num_workers,
        )
    test_loader =  wds.WebLoader(
            val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.num_workers,
        )

    return train_loader, test_loader
