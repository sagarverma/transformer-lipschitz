from webdataset import ShardWriter
from threading import Thread, Lock


class TarThread(Thread):
    def __init__(self, sample, tarstream, lock: Lock, *args, **kwargs):
        super(TarThread, self).__init__(*args, **kwargs)
        self.sample = sample
        self.tarstream = tarstream
        self.lock = lock
        self.size = 0
    def run(self):
        self.lock.acquire()
        for k in self.sample.keys():
            self.tarstream.addfile(
                self.sample[k]['info'], 
                self.sample[k]['stream']
                )
        self.lock.release()
        return True


class ShardWriterCustom(ShardWriter):

    def __init__(self, pattern, maxcount, **kw):
        super(ShardWriterCustom,self).__init__(pattern, maxcount=maxcount, **kw)
        self.lock = Lock()
        self.threads = []
    
    def write(self,objs):
        """Write a sample.
        :param obj: sample to be written
        """
        for obj in objs:
            if self.tarstream is None or self.count >= self.maxcount:
                while len(self.threads) > 0:
                    t = self.threads.pop(0)
                    t.join()
                self.next_stream()
            t = TarThread(sample = obj, tarstream = self.tarstream.tarstream, lock= self.lock)
            t.start()
            self.threads.append(t)
            sz = 0
            for key_ in obj.keys():
                sz += obj[key_]['info'].size
            self.count += 1
            self.total += 1
            self.size += sz
    
    def close(self):
        while len(self.threads) > 0:
            t = self.threads.pop(0)
            t.join()
        return super().close()
