from PIL import Image
import os
import torch
import json

from torchvision.datasets import ImageFolder
from datasets.imagenet_constants import indices_in_1k

class ImageNet(ImageFolder):
    """
    Simply the Imagenet dataset reformatted so that is matches the other dataset outputs.
    EDIT: include support for Imagenet-A
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        if len(self.classes) == 200:
            self.map = indices_in_1k
        else:
            self.map = list(range(1000))
        
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, self.map[label], 0, idx

"""
PLEASE_NOTE: this code is for the validation set ONLY.

Source of real.json and parsing code:
https://github.com/google-research/reassessed-imagenet
"""

class ImagenetVal:
    def __init__(self, root, transform=None):
        self.root = os.path.join(root, "val")
        self.transform = transform

        with open("datasets/imagenetval.json") as real_labels:
            real_labels = json.load(real_labels)
            real_labels = [
                [f'ILSVRC2012_val_{i + 1:08d}.JPEG', labels] 
                for i, labels in enumerate(real_labels)
                if labels
            ]
        self.metadata = real_labels
    
    def __getitem__(self, idx):
        img_name, label = self.metadata[idx]
        img = Image.open(os.path.join(self.root, img_name)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # add img_name to return image name as well. 
        return img, label, 0, idx
    
    def __len__(self):
        return len(self.metadata)

class ImagenetA(ImageFolder):
    """
    Imagenet-A dataset
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.map = indices_in_1k

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, 0, idx