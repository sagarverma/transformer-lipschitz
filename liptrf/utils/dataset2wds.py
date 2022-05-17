import os
import math
import pickle
import json
import tqdm
import time
import numpy as np

import webdataset as wds
from webdataset import ShardWriter
from torch.utils.data import Dataset, Subset, DataLoader
from concurrent.futures import ProcessPoolExecutor as Pool
from stream_consumer2tar import ShardWriterCustom
from tar_stream_ds import StreamDataset
from helpers import collate


def toHHMMSS(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return "%d:%02d:%02d" % (hour, minutes, seconds)


class Dataset2WebDataset:
    r"""
    Creates Posix Tar shards for DataStore(datastore.granular.ai) for torch.utils.data.Dataset
    Here, key_names should be in
        **class** :         ["cls", "cls2", "class", "count", "index", "inx", "id"]
    
        **text** :          ["txt", "text", "transcript"]
    
        **image** :         ["png", "jpg", "jpeg", "img", "image", "pbm", "pgm", "ppm"]
        **image_path** :    ["path"]
        **bytes** :         ["bytes"]
    
        **pickle_object** : ["pyd", "pickle"]
    
        **torch_object** :  ["pth"]
    
        **numpy.ndarray** : ["npy"]  don't use not stable
    
        **json_dict** :     ["json", "jsn"]
    Parameters
    ----------
    dataset : `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
        Dataset to convert
    key_names : `list <https://docs.python.org/3/tutorial/introduction.html#lists>`_
        Ordered list for the items returned by the input dataset iterator
    transforms : `object <https://docs.python.org/3/reference/datamodel.html#objects-values-and-types>`_
        Transforms object to be saved on datastore for the input dataset
    mode : `str <https://docs.python.org/3/library/stdtypes.html#str>`_
        Dataset mode type = train/val/test
    shard_size : `int <https://docs.python.org/3/library/functions.html#int>`_
        Upper bound on shard memory size in GBs, default = 10GB
    Examples
    --------
    Please check `here <https://github.com/granularai/phobos/blob/develop/examples/dataset2wds/Example.ipynb>`_ 
    """
    def __init__(self, 
            dataset: Dataset, 
            keys: list, 
            transforms: object, 
            mode:str = 'train', 
            shard_size = 10,
            shard_size_bytes=None
            ) -> None:
        self.keys = keys
        self.dataset = dataset
        self.mode = mode
        self.transforms = transforms
        self.index_len = int(math.log10(len(self.dataset)))+1 if len(self.dataset) > 0 else 1
        
        if not shard_size_bytes:
            shard_size = Dataset2WebDataset.shardSizeToBytes(shard_size)
        else:
            shard_size = shard_size_bytes
        sz_,self.time_per_shard = Dataset2WebDataset.getSampleSize(self.dataset, self.keys, self.index_len)
        self.sample_size = sz_
        self.samples_per_shard = int(shard_size // sz_)
        self.client = None
        self.dataset = StreamDataset(self.dataset,keys,self.index_len)

        print(f"samples_per_shard:{self.samples_per_shard} sample_size_bytes:{self.sample_size}, ETA: {toHHMMSS(len(self.dataset)*self.time_per_shard)}")

    @staticmethod
    def getBytes(path):
        with open(path, 'rb') as fp:
            return fp.read()

    @staticmethod
    def getBytesLen(stream):
        return len(stream)
    
    @staticmethod
    def shardSizeToBytes(shard_size):
        shard_dec = shard_size - shard_size//1
        shard_dec = math.ceil(shard_dec*1000)
        shard_dec = (shard_dec << 10) << 10
        shard_size = int(shard_size)
        shard_size = ((shard_size//1 << 10) << 10) << 10
        shard_size += shard_dec
        return shard_size

    @staticmethod
    def getSampleSize(dataset, keys, index_len):
        index = 1
        time_elapsed = time.time()
        items = dataset[0]
        with ShardWriter('tmp-%01d.tar', maxcount=1) as sink:
            sample = {
                "__key__": str(index).zfill(index_len)
            }
            for key, val in zip(keys, items):
                if type(val) == str and key.split('.')[-1] == "path":
                    val = Dataset2WebDataset.getBytes(val)
                sample[key] = val
            sink.write(sample)
        time_elapsed = time.time()-time_elapsed
        with open("tmp-0.tar", 'rb') as fp:
            fp.seek(0, os.SEEK_END)
            sz_ = fp.tell()
        os.remove("tmp-0.tar")     
        return sz_, time_elapsed
    
    @staticmethod
    def getShardSize(dataset,nodes,keys,max_shard_size):
        max_shard_size_bytes = Dataset2WebDataset.shardSizeToBytes(max_shard_size)
        index_len = int(math.log10(len(dataset)))+1 if len(dataset) > 0 else 1
        sample_sz,_ = Dataset2WebDataset.getSampleSize(dataset, keys, index_len)
        dataset_samples_sz = (len(dataset)*sample_sz) // nodes
        if dataset_samples_sz >= max_shard_size_bytes:
            res = Dataset2WebDataset.shardSizeToBytes(max_shard_size)
        else:
            res = dataset_samples_sz
        return res

    def getTransform(self):
        return self.transform
    
    def getMetadata(self, num_shards):
        
        return {
                "transforms": f"transforms_{self.mode}.pkl",
                "url_posix_path": self.mode+"-{"+str(0).zfill(
                    self.shards_len)+".."+str(num_shards-1).zfill(self.shards_len)+"}.tar",
                "mode": self.mode,
                "shards_count": num_shards,
                "num_samples": len(self.dataset),
                "keys": self.keys,
                "samples_per_shard": self.samples_per_shard
                }
    def writeSingleShard(self, obj):
        shards_out = obj['shards_out']
        start_shard = obj['num']
        shard_loader = obj['shard']
        print(f"writing {start_shard}")
        with ShardWriterCustom(
            pattern = shards_out, 
            maxcount = self.samples_per_shard,
            start_shard = start_shard) as sink:
            progress_bar = tqdm.tqdm(shard_loader)
            print(shard_loader.dataset.__len__())
            for samples in progress_bar:
                sink.write(samples)
        if self.client:
            fl_ = f"{self.mode}-{str(start_shard).zfill(self.shards_len)}.tar"
            local_path = os.path.join(
                os.path.join(os.curdir,"tmp"),
                fl_
            )
            cnt = 0
            max_tries = obj['max_tries']
            while True:
                print("starting write")
                res = self.client.putObject(local_path, fl_)
                print("written")
                print(res)
                if res:
                    os.remove(local_path)
                    break
                elif cnt == max_tries-1:
                    exit(0)
                cnt += 1
        print(f"completed {start_shard}")

            
                #progress_bar.set_description(f"Writing {self.mode} shards, In progress : {(index+1)}/{dataset_ln}, ETA: {toHHMMSS((dataset_ln-index-1)*self.time_per_shard)}")

    def getShards(self, shuffle, out_dir, batch_size, num_workers, bucket_path="", max_tries=3):
        dataset_ln = len(self.dataset)
        
        # process pool
        if shuffle:
            data_ind = np.random.permutation(dataset_ln).tolist()
        else:
            data_ind = list(range(dataset_ln))
        shards_ind = [ [i,i+self.samples_per_shard] for i in range(0, dataset_ln, self.samples_per_shard)]
        if shards_ind[-1][1] >= dataset_ln:
            shards_ind[-1][1] = dataset_ln

        num_shards = len(shards_ind)

        self.shards_len = int(math.log10(num_shards))+1 if num_shards > 0 else 1

        shards_out = os.path.join(out_dir,f"{self.mode}-%0{self.shards_len}d.tar")

        shards = [
            {
            'shard': DataLoader(
                    Subset(
                        self.dataset,
                        data_ind[shards_ind[i][0]:shards_ind[i][1]]
                    ),
                    batch_size = batch_size,
                    num_workers = num_workers,
                    collate_fn=collate
                ),
            'num': i,
            'shards_out': shards_out,
            'max_tries': max_tries
            } for i in range(len(shards_ind))
        ]

        return shards
        


    def writeShards(self, out_path, provider='local', shuffle=True, batch_size = 10, num_workers=10, pool_size=5, max_tries=3):
        r"""
        Writes shards on local filesystem
        Parameters
        ----------
        out_path : `str <https://docs.python.org/3/library/stdtypes.html#str>`_
            Output path for writing shards
            local, gcp
        """

        bucket_path, out_dir = None, None

        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_dir = out_path
        transform_out = os.path.join(out_dir, f"transforms_{self.mode}.pkl")
        metadata_out = os.path.join(out_dir, f"metadata_{self.mode}.json")


        shards = self.getShards(
            shuffle, 
            out_dir, 
            batch_size, 
            num_workers,
            bucket_path=out_path,
            max_tries=max_tries
            )

        with Pool(max_workers=pool_size) as pool:
            pool.map(self.writeSingleShard,shards)

        #for shard in shards:
        #    self.writeSingleShard(shard)

        metadata = self.getMetadata(len(shards))
        

        if provider == 'local':
            if self.transforms:
                with open(transform_out,'wb') as fp:
                    pickle.dump(self.transforms,fp)
            with open(metadata_out, 'w') as fp:
                json.dump(metadata, fp)
        else:
            if self.transforms:
                for i in range(max_tries):
                    resp = self.client.putObjectStream(pickle.dumps(self.transforms),transform_out)
                    if resp == 200:
                        break
            for i in range(max_tries):
                resp = self.client.putObjectStream(json.dumps(metadata).encode("utf-8"),metadata_out)
                if resp == 200:
                    break
    

    

    @classmethod
    def trainVal2Shards(
        cls,
        train_dataset,
        val_dataset,
        keys,
        distributed_val=True,
        mode='node',
        val=16,
        max_shard_size=2,
        shuffle=True,
        out_path='./',
        provider='local',
        batch_size = 10,
        num_workers = 10,
        pool_size = 5,
        train_transform=None,
        val_transform=None
    ):
        '''mode = node/size'''
        if mode == 'node':
            val = int(val)

      
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            return
        else:
            paths = []
            for path in ['train','val']:
                tmp = os.path.join(out_path,path)
                if not os.path.exists(tmp):
                    os.makedirs(tmp)
                paths.append(tmp)
            train_path, val_path = paths
        
        if mode == 'node':
            nodes = val
            assert len(train_dataset) >= nodes and len(val_dataset) >= nodes
            train_shard_size = Dataset2WebDataset.getShardSize(train_dataset,nodes,keys,max_shard_size)
            if distributed_val:
                val_shard_size = Dataset2WebDataset.getShardSize(val_dataset,nodes,keys,max_shard_size)
            else:
                val_shard_size = Dataset2WebDataset.shardSizeToBytes(max_shard_size)
        else:
            train_shard_size = Dataset2WebDataset.shardSizeToBytes(val)
            val_shard_size = Dataset2WebDataset.shardSizeToBytes(val)

        train = cls(
            train_dataset,
            keys,
            mode='train',
            shard_size_bytes=train_shard_size,
            transforms=train_transform
            )
        val = cls(
            val_dataset,
            keys,
            mode='val',
            shard_size_bytes=val_shard_size,
            transforms=val_transform
            )
        print("writing train shards")
        train.writeShards(
            shuffle=shuffle, 
            out_path=train_path,
            provider=provider,
            batch_size=batch_size,
            num_workers=num_workers,
            pool_size=pool_size
            )
        print("writing val shards")
        val.writeShards(
            shuffle=False,
            out_path=val_path,
            provider=provider,
            batch_size=batch_size,
            num_workers=num_workers,
            pool_size=pool_size
            )
        print("Completed!")