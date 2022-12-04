from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
from utils import get_counts

class Waterbirds:

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, split='train', transform=transform):
        splits = ['train', 'val', 'test']
        root = os.path.join(root, 'waterbird_1.0_forest2water2')
        self.root = root
        self.split = split
        self.transform = transform
        self.df = pd.read_csv(os.path.join(root, 'metadata.csv'))
        self.df = self.df[self.df['split'] == splits.index(split)]
        self.filenames = [os.path.join(root, f) for f in self.df['img_filename']]
        self.labels = self.df['y'].values
        self.groups = []
        for y, p in zip(self.df['y'], self.df['place']):
            if y == 0 and p == 0:
                self.groups.append(0)
            elif y == 0 and p == 1:
                self.groups.append(1)
            elif y == 1 and p == 0:
                self.groups.append(2)
            elif y == 1 and p == 1:
                self.groups.append(3)
        self.samples = list(zip(self.filenames, self.labels, self.groups))
        self.class_weights = get_counts(self.labels)
        self.classes = ['landbird', 'waterbird']
        self.class_sizes = [len(self.df[self.df['y'] == 0]), len(self.df[self.df['y'] == 1])]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename, label, group = self.samples[idx]
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        return img, label, group