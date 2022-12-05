from PIL import Image
import os
import torch
import json
import pandas as pd

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

imagenet_a_wnids = ['n01498041', 'n01531178', 'n01534433', 'n01558993', 'n01580077', 'n01614925', 'n01616318', 'n01631663', 'n01641577', 'n01669191', 'n01677366', 'n01687978', 'n01694178', 'n01698640', 'n01735189', 'n01770081', 'n01770393', 'n01774750', 'n01784675', 'n01819313', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01882714', 'n01910747', 'n01914609', 'n01924916', 'n01944390', 'n01985128', 'n01986214', 'n02007558', 'n02009912', 'n02037110', 'n02051845', 'n02077923', 'n02085620', 'n02099601', 'n02106550', 'n02106662', 'n02110958', 'n02119022', 'n02123394', 'n02127052', 'n02129165', 'n02133161', 'n02137549', 'n02165456', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02259212', 'n02268443', 'n02279972', 'n02280649', 'n02281787', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02361337', 'n02410509', 'n02445715', 'n02454379', 'n02486410', 'n02492035', 'n02504458', 'n02655020', 'n02669723', 'n02672831', 'n02676566', 'n02690373', 'n02701002', 'n02730930', 'n02777292', 'n02782093', 'n02787622', 'n02793495', 'n02797295', 'n02802426', 'n02814860', 'n02815834', 'n02837789', 'n02879718', 'n02883205', 'n02895154', 'n02906734', 'n02948072', 'n02951358', 'n02980441', 'n02992211', 'n02999410', 'n03014705', 'n03026506', 'n03124043', 'n03125729', 'n03187595', 'n03196217', 'n03223299', 'n03250847', 'n03255030', 'n03291819', 'n03325584', 'n03355925', 'n03384352', 'n03388043', 'n03417042', 'n03443371', 'n03444034', 'n03445924', 'n03452741', 'n03483316', 'n03584829', 'n03590841', 'n03594945', 'n03617480', 'n03666591', 'n03670208', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03775071', 'n03788195', 'n03804744', 'n03837869', 'n03840681', 'n03854065', 'n03888257', 'n03891332', 'n03935335', 'n03982430', 'n04019541', 'n04033901', 'n04039381', 'n04067472', 'n04086273', 'n04099969', 'n04118538', 'n04131690', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04179913', 'n04208210', 'n04235860', 'n04252077', 'n04252225', 'n04254120', 'n04270147', 'n04275548', 'n04310018', 'n04317175', 'n04344873', 'n04347754', 'n04355338', 'n04366367', 'n04376876', 'n04389033', 'n04399382', 'n04442312', 'n04456115', 'n04482393', 'n04507155', 'n04509417', 'n04532670', 'n04540053', 'n04554684', 'n04562935', 'n04591713', 'n04606251', 'n07583066', 'n07695742', 'n07697313', 'n07697537', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07749582', 'n07753592', 'n07760859', 'n07768694', 'n07831146', 'n09229709', 'n09246464', 'n09472597', 'n09835506', 'n11879895', 'n12057211', 'n12144580', 'n12267677']

class Imagenette(ImageFolder):

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, cfg, transform=transform, split='train', type='imagenette2'):
        s = 'train' if split == 'train' else 'val'
        super().__init__(os.path.join(root, type, s), transform)
        self.split = split
        self.cfg = cfg
        if split == 'val':
            sample_idxs = [i for i in range(len(self.samples)) if i % 2 == 0]
        elif split == 'test':
            sample_idxs = [i for i in range(len(self.samples)) if i % 2 == 1]
        else:
            sample_idxs = [i for i in range(len(self.samples))]
        self.samples = [self.samples[i] for i in sample_idxs]
        self.labels = [s[1] for s in self.samples]

class ImagenetteWoof(Imagenette):

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, cfg, transform=transform, split='train', type='imagewoof2'):
        super().__init__(root, cfg, transform, split, type)

class NoisyImagenette(Imagenette):

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, root, cfg, transform=transform, split='train', type='imagenette2'):
        super().__init__(root, cfg=cfg, transform=transform, type=type)
        assert self.cfg.noise.p in [0, 0.01, 0.05, 0.25, 0.50], "Noise level not supported"
        self.df = pd.read_csv(os.path.join(root, 'noisy_imagenette.csv'))
        is_valid = True if split == 'val' else False
        self.df = self.df[self.df.is_valid == is_valid]
        self.labels = self.df[f"noisy_labels_{self.cfg.noise.p}"].tolist()
        self.image_paths = [path.replace('val/', '').replace('train/', '') for path in self.df['path']]
        self.samples = list(zip(self.df['path'].tolist(), self.labels))

class ImagenetteC(ImageFolder):

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def __init__(self, root, cfg, transform=transform, split='test', corruption='glass_blur', severity=1):
        super().__init__(os.path.join(root, 'imagenette-c', corruption, str(severity)), transform=transform)
        self.cfg = cfg
        self.labels = [s[1] for s in self.samples]

# class ImagenetteA(ImageFolder):
#     """
#     Subset of Imagenet-a that contains only the classes in imagenette.
#     """
#     transform = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#     def __init__(self, root, cfg, transform=transform, split='train'):
#         assert split in ['train', 'val'], "Imagenette does not have a train/val set"
#         super().__init__(root, transform)
#         self.classes = ['n03000684','n03425413','n03394916','n02102040','n02979186','n01440764','n03445777','n03028079']
#         self.class_intersection = ['n03417042', 'n03888257']
#         self.cfg = cfg
#         # map labels to 0-9
#         self.class_to_idx = {i: }
