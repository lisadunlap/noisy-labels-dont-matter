import torch 
import numpy as np

from datasets.waterbirds import Waterbirds
from datasets.imagenette import Imagenette, ImagenetteWoof, NoisyImagenette, ImagenetteC
from datasets.base import CombinedDataset, SubsetDataset

def get_dataset(name, cfg, split='train'):
    """
    Gets a dataset by name
    """
    if name == 'Imagenette':
        return Imagenette(root=cfg.data.root, cfg=cfg, split=split)
    elif name == 'ImagenetteWoof':
        return ImagenetteWoof(root=cfg.data.root, cfg=cfg, split=split)
    elif name == 'Waterbirds':
        return Waterbirds(root=cfg.data.root, split=split)
    elif name == 'NoisyImagenette':
        return NoisyImagenette(root=cfg.data.root, cfg=cfg, split=split)
    elif name == 'ImagenetteC':
        corruption=cfg.data.imagenetc_corruption
        # datasets = [ImagenetteC(root=cfg.data.root, cfg=cfg, split=split, corruption=corruption, severity=i) for i in range(1,6)]
        return ImagenetteC(root=cfg.data.root, cfg=cfg, split=split, corruption=corruption, severity=1)
