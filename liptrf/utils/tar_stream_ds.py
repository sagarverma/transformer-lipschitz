import tarfile
import io
from torch.utils.data import Dataset
from liptrf.utils.helpers import make_encoder




class StreamDataset(Dataset):
    
    def __init__(self, dataset, keys, index_len):
        super().__init__()
        self.index_len = index_len
        self.keys = keys
        self.dataset = dataset
        self.encoder = make_encoder(True)
        self.mode = 0o0444
        self.user = "granular"
        self.group = "granular"
        self.compress = False

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        sample = {
            "__key__" : str(index).zfill(self.index_len)
        }
        for key,val in zip(self.keys,item):
            #val=val[0]
            if type(val) == str and key.split('.')[-1] == "path":
                with open(val, 'rb') as fp:
                    val = fp.read()
            sample[key] = val
        #creating stream of sample
        obj = self.encoder(sample)
        
        if "__key__" not in obj:
            raise ValueError("object must contain a __key__")
        
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(f"{k} doesn't map to a bytes after encoding ({type(v)})")
        
        key = obj["__key__"]
        out = {}

        for k in sorted(obj.keys()):
            if k == "__key__":
                continue
            if k[0] == "_":
                continue
            v = obj[k]
            if isinstance(v, str):
                v = v.encode("utf-8")
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mode = self.mode
            ti.uname = self.user
            ti.gname = self.group
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(f"converter didn't yield bytes: {k}, {type(v)}")
            stream = io.BytesIO(v)
            out[k] = {
                'info': ti,
                'stream': stream
                }
        return out    
        
        