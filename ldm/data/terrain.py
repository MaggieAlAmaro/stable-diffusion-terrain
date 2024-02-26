import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths
# from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class TerrainBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None
        self.augment = kwargs.get('augment', False)
        self.transform = self.getAugmentTransform() if self.augment else None
            

    def getAugmentTransform(self):
        transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1),
                albumentations.VerticalFlip(p=1),
            ], p=0.5),
            albumentations.RandomRotate90(p=0.5),
            ]
        )
        return transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        if self.transform:
            example['image'] = self.transform(image=example['image'])['image']  #CHECK IF NEW EXAMPLE IMAGE IS DIFFERENT THAN ORIGINAL
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex



class TerrainRGBATrain(TerrainBase):
    def __init__(self, size,augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/RGBAv4_NewExpMean_FullData"
        with open("data/RGBA_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)#, grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class TerrainRGBAValidation(TerrainBase):
    def __init__(self, size, augment= None,keys=None):
        super().__init__(augment=augment)
        root = "data/RGBAv4_NewExpMean_FullData"
        with open("data/RGBA_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)#,grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class TerrainRGBATest(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/RGBAv4_NewExpMean_FullData"
        with open("data/RGBA_test.txt", "r") as f:
        # root = "data/terrainTrain"
        # with open("data/newEurope_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)#,grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys




####### TEST SET
        
        
class TestRGBATrain(TerrainBase):
    def __init__(self, size,augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainTinyRGBA"
        with open("data/tinyRGBA_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)#, grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class TestRGBAValidation(TerrainBase):
    def __init__(self, size, augment= None,keys=None):
        super().__init__(augment=augment)
        root = "data/terrainTinyRGBA"
        with open("data/tinyRGBA_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)#,grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys

class TestRGBATest(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainTinyRGBA"
        with open("data/tinyRGBA_test.txt", "r") as f:
        # root = "data/terrainTrain"
        # with open("data/newEurope_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)#,grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys




class TerrainGSTrain(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainGS"
        with open("data/terrainGS_train.txt", "r") as f:
        # root = "data/newEurope"
        # with open("data/newEurope_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class TerrainGSValidation(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainGS"
        with open("data/terrainGS_validation.txt", "r") as f:
        # root = "data/terrainTrain"
        # with open("data/newEurope_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False,grayscale=True)
        #self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys




