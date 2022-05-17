import glob

from liptrf.utils.dataset2wds import Dataset2WebDataset 
from torch.utils.data import Dataset 


class TinyImageNet(Dataset):
    def __init__(self, meta, class_map):
        self.meta = meta
        self.class_map = class_map

    def __len__(self):
        return len(self.meta)

    
    def __getitem__(self, idx):
        return self.meta[idx], self.class_map[idx.split('/')[-2]]


fin = open('./data/tiny-imagenet-200/words.txt', 'r')
class_map = {}
i = 0
for line in fin.readlines():
    class_map[line.split('\t')] = i 
    i += 1

train_samples = glob.glob('./data/tiny-imagenet-200/train/**/*.JPEG')
val_samples = glob.glob('./data/tiny-imagenet-200/val/**/*.JPEG')

train_dataset = TinyImageNet(train_samples)
val_dataset = TinyImageNet(val_samples)

keys=["jpeg.path", "_class"]

train_exporter = Dataset2WebDataset(dataset=train_dataset, 
                                keys=keys, 
                                transforms=None,
                                shard_size=0.1,
                                mode='train')
val_exporter = Dataset2WebDataset(dataset=val_dataset, 
                                keys=keys, 
                                transforms=None,
                                shard_size=0.1,
                                mode='val')


train_exporter.writeShards(out_path='./data/tiny-imagenet-200/shards/train', shuffle=False, batch_size=20, num_workers=20, pool_size=1)
val_exporter.writeShards(out_path='./data/tiny-imagenet-200/shards/val', shuffle=False, batch_size=20, num_workers=20, pool_size=1)