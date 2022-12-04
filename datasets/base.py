import torch
import pandas as pd
from utils import get_counts
import methods
from PIL import Image
from torchvision.datasets import ImageFolder

class BaseDataset:
    """
    Wrapper for dataset class that allows for mislables.
    """
    def __init__(self, dataset, cfg, clean=False):
        self.dataset = dataset
        self.cfg = cfg
        self.clean = clean
        # clean = True overrides the config
        self.mislabel_method = getattr(methods, 'noop' if clean else cfg.noise.method)
        self.class_weights = get_counts(self.dataset.labels)
        self.clean_labels = self.dataset.labels
        self.labels, self.noisy_idxs = self.mislabel_method(self.dataset.labels, self.cfg.noise.p)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = self.labels[idx]

        if len(item) == 2:
            return item[0], label, 0, idx
        elif len(item) == 3:
            return item[0], self.labels[idx], item[2], idx
        else:
            return item[0], self.labels[idx], item[2], idx

class CombinedDataset:
    """
    Wrapper that combines two datasets into one
    """
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.samples = self.dataset1.samples + self.dataset2.samples
        self.dataset_idx = [0] * len(self.dataset1) + [1] * len(self.dataset2)
        self.idxs = list(range((len(self.dataset1)))) + list(range((len(self.dataset2))))
        if len(self.samples[0]) == 2:
            self.filenames, self.labels = zip(*[(s[0], s[1]) for s in self.samples])
            self.groups = [0] * len(self.dataset1) + [1] * len(self.dataset2)
        else:
            self.filenames, self.labels, self.groups = zip(*[(s[0], s[1], s[2]) for s in self.samples])
        self.class_weights = get_counts(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        didx = self.dataset_idx[idx]
        dataset = self.dataset1 if self.dataset_idx[idx] == 0 else self.dataset2
        item = dataset[didx]
        if len(item) == 2:
            return item[0], item[1], self.groups[idx], idx
        elif len(item) == 3:
            return item[0], item[1], item[2], idx
        else:
            return item[0], item[1], item[2], idx

class SubsetDataset:
    """
    Wrapper that subsets a dataset
    """
    def __init__(self, dataset, classes, transform=None):
        if type(classes[0]) != int:
            classes = [dataset.classes.index(c) for c in classes]
        print(classes)
        self.samples = [item for item in dataset.samples if item[1] in classes]
        self.labels = [item[1] for item in self.samples]
        self.transform = transform
        self.classes = [dataset.classes[c] for c in classes]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item[0])
        if self.transform:
            img = self.transform(img)
        if len(item) == 2:
            return item[0], item[1], 0, idx
        elif len(item) == 3:
            return item[0], item[1], item[2], idx
        return item