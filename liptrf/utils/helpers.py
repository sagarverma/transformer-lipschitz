import io
import pickle
import re
import json

import numpy as np
from . import tenbin


def imageencoder(image, format="PNG"):  # skipcq: PYL-W0622
    """Compress an image using PIL and return it as a string.
    Can handle float or uint8 images.
    :param image: ndarray representing an image
    :param format: compression format (PNG, JPEG, PPM)
    """
    import PIL
    if isinstance(image, np.ndarray):
        if image.dtype in [np.dtype("f"), np.dtype("d")]:
            if not (np.amin(image) > -0.001 and np.amax(image) < 1.001):
                raise ValueError(
                    f"image values out of range {np.amin(image)} {np.amax(image)}"
                )
            image = np.clip(image, 0.0, 1.0)
            image = np.array(image * 255.0, "uint8")
        image = PIL.Image.fromarray(image)
    if format.upper() == "JPG":
        format = "JPEG"
    elif format.upper() in ["IMG", "IMAGE"]:
        format = "PPM"
    if format == "JPEG":
        opts = dict(quality=100)
    else:
        opts = {}
    with io.BytesIO() as result:
        image.save(result, format=format, **opts)
        return result.getvalue()


def bytestr(data):
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("ascii")
    return str(data).encode("ascii")


def torch_dumps(data):
    import io
    import torch

    stream = io.BytesIO()
    torch.save(data, stream)
    return stream.getvalue()

def encode_ascii(x):
    return str(x).encode("ascii")

def encode_text(x):
    return x.encode("utf-8")

def encode_json(x):
    return json.dumps(x).encode("utf-8")

class ImageInc:
    def __init__(self, extension):
        self.extension = extension
    def __call__(self, x):
        return imageencoder(x, self.extension)
        
def tenbinf(x):  # skipcq: PYL-E0102
    if isinstance(x, list):
        return memoryview(tenbin.encode_buffer(x))
    else:
        return memoryview(tenbin.encode_buffer([x]))

def make_handlers():
    handlers = {}
    for extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        handlers[extension] = encode_ascii
    for extension in ["txt", "text", "transcript"]:
        handlers[extension] = encode_text
    for extension in ["png", "jpg", "jpeg", "img", "image", "pbm", "pgm", "ppm"]:
        handlers[extension] = ImageInc(extension)
    for extension in ["pyd", "pickle"]:
        handlers[extension] = pickle.dumps
    for extension in ["pth"]:
        handlers[extension] = torch_dumps
    for extension in ["json", "jsn"]:
        handlers[extension] = encode_json
    for extension in ["ten", "tb"]:

        handlers[extension] = tenbinf
    try:
        import msgpack

        for extension in ["mp", "msgpack", "msg"]:
            handlers[extension] = msgpack.packb
    except ImportError:
        pass
    return handlers


default_handlers = {"default": make_handlers()}


def encode_based_on_extension1(data, tname, handlers):
    if tname[0] == "_":
        if not isinstance(data, str):
            raise ValueError("the values of metadata must be of string type")
        return data
    extension = re.sub(r".*\.", "", tname).lower()
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    handler = handlers.get(extension)
    if handler is None:
        raise ValueError(f"no handler found for {extension}")
    return handler(data)


def encode_based_on_extension(sample, handlers):
    res = {
        k: encode_based_on_extension1(v, k, handlers) for k, v in list(sample.items())
    }
    return res

def collate(data):
    return data

class make_encoder:
    def __init__(self,spec):
        self.spec = spec
        if spec is False or spec is None:
            self.func = self.enc1
        elif callable(spec):
            self.func = spec
        elif isinstance(spec, dict):
            self.func = self.enc2

        elif isinstance(spec, str) or spec is True:
            if spec is True:
                spec = "default"
            handlers = default_handlers.get(spec)
            if handlers is None:
                raise ValueError(f"no handler found for {spec}")
            self.handlers = handlers
            self.func = self.enc3
        else:
            raise ValueError(f"{spec}: unknown decoder spec")
        if not callable(self.func):
            raise ValueError(f"{spec} did not yield a callable encoder")
    def enc1(self,x):
        return x
    def enc2(self,sample):
        return encode_based_on_extension(sample, self.spec)
    def enc3(self, sample):
        return encode_based_on_extension(sample, self.handlers)
    def __call__(self,x):
        return self.func(x)