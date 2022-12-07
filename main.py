import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import copy

from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

import omegaconf
from omegaconf import OmegaConf
import argparse
import wandb

from utils import read_unknowns, nest_dict, flatten_config
from datasets.get_dataset import get_dataset
from datasets.base import BaseDataset
from utils import evaluate
import methods

parser = argparse.ArgumentParser(description='Train/Val')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
# flags = parser.parse_args()
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not args.exp.wandb:
    os.environ['WANDB_SILENT']="true"

run = f"{args.exp.run}-debug" if args.exp.debug else args.exp.run
wandb.init(entity='lisadunlap', project='noisy_labels_dont_matter', name=run, config=flatten_config(args))

train_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='train'), args)
val_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='val'), args, clean=args.noise.clean_val)
test_set = BaseDataset(get_dataset(args.data.test_dataset, cfg=args, split='test'), args, clean=True)
# clean_val_set = BaseDataset(get_dataset(args.data.dataset, cfg=args, split='val'), args, clean=True)
weights = train_set.class_weights

labels = np.array(train_set.labels) == np.array(train_set.clean_labels)
p = args.noise.p if args.noise.method != 'noop' else 0
print(f"Training {args.exp.run} on {args.data.dataset} with {p*100}% {args.noise.method} noise ({len(labels[labels == False])}/{len(labels)})")

pretrained_weights =ResNet50_Weights.IMAGENET1K_V2 if args.model.ft else ResNet50_Weights.DEFAULT
model = resnet50(weights=pretrained_weights).cuda()
num_classes, dim = model.fc.weight.shape
model.fc = nn.Linear(dim, len(train_set.classes)).cuda()
for name, param in model.named_parameters():
    if "fc" in name or not args.model.ft:
        param.requires_grad = True
    else:
        param.requires_grad = False
model = nn.DataParallel(model).cuda()


trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.data.batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(
        val_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)
# clean_val_loader = torch.utils.data.DataLoader(
#         clean_val_set, batch_size=args.data.batch_size, shuffle=False, num_workers=2)

class_criterion = nn.CrossEntropyLoss(weight=weights.cuda())
m = nn.Softmax(dim=1)
optimizer = torch.optim.SGD(model.parameters(), lr=args.hps.lr, weight_decay=args.hps.weight_decay, momentum=0.9)

def train_val_loop(loader, epoch, phase="train", best_acc=0):
    """
    One epoch of train-val loop.
    Returns of dict of metrics to log
    """
    if phase == "train":
        model.train()
    else:
        model.eval()
    total_loss, cls_correct, total = 0, 0, 0
    cls_true, cls_pred, cls_groups, dom_true = np.array([]), np.array([]), np.array([]), np.array([])
    with torch.set_grad_enabled(phase == 'train'):
        with tqdm(total=len(loader), desc=f'{phase} epoch {epoch}') as pbar:
            for i, (inp, cls_target, cls_group, idx) in enumerate(loader):
                inp, cls_target = inp.cuda().float(), cls_target.cuda().long()
                if phase == "train":
                    optimizer.zero_grad()
                out = model(inp)
                conf, pred = torch.max(m(out), dim=-1)
                cls_loss = class_criterion(out.float(), cls_target)
                loss = cls_loss 
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                total_loss += cls_loss.item()
                total += cls_target.size(0)
                cls_correct += pred.eq(cls_target).sum().item()

                cls_true = np.append(cls_true, cls_target.cpu().numpy())
                cls_pred = np.append(cls_pred, pred.cpu().numpy())
                cls_groups = np.append(cls_groups, cls_group.cpu().numpy())
                pbar.update(i)
            
    accuracy, balanced_acc, class_accuracy, group_accuracy =  evaluate(cls_pred, cls_true, cls_groups)

    wandb.log({f"{phase} loss": total_loss, f"{phase} cls acc": accuracy, f"{phase} balanced class acc": balanced_acc, 
                f"{phase} class acc": class_accuracy, f"{phase} group acc": group_accuracy, "epoch": epoch})

    if phase == 'val' and balanced_acc > best_acc:
        best_acc = balanced_acc
        if (not args.exp.debug) and (not args.noise.clean_val) and (args.model.resume_epoch == 'best'):
            save_checkpoint(model, balanced_acc, group_accuracy, epoch, best=True)
        wandb.summary[f'best {phase} acc'] = balanced_acc
        wandb.summary[f'best {phase} group acc'] = group_accuracy
        wandb.summary[f'best {phase} epoch'] = epoch
    elif phase == 'val' and epoch % args.model.save_every == 0:
        if (not args.exp.debug) and (not args.noise.clean_val) and (args.model.resume_epoch == 'best'):
            save_checkpoint(model, balanced_acc, group_accuracy, epoch)
    elif phase == 'test':
        wandb.summary[f'{phase} acc'] = balanced_acc
        wandb.summary[f'{phase} group acc'] = group_accuracy
    return best_acc if phase == 'val' else balanced_acc


def save_checkpoint(model, acc, group_accuracy, epoch, best=False):
    state = {
        "acc": acc,
        "group_acc": group_accuracy,
        "epoch": epoch,
        "net": model.module.state_dict()
    }
    checkpoint_dir = f'./checkpoint/{args.data.dataset}/{args.exp.run}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if best:
        checkpoint_path = f'{checkpoint_dir}/model_best.pth'
        torch.save(state, f'{checkpoint_dir}/model_best.pth')
        wandb.save(f'{checkpoint_dir}/model_best.pth')
    else:
        checkpoint_path = f'{checkpoint_dir}/model_{epoch}.pth'
    print(f'Saving checkpoint with acc {acc} to {checkpoint_path}')
    torch.save(state, f'{checkpoint_dir}/model_{epoch}.pth')
    wandb.save(f'{checkpoint_dir}/model_{epoch}.pth')

def get_model_path(epoch=-1):
    if epoch == -1:
        path = f'./checkpoint/{args.data.dataset}/{args.exp.run}/model_best.pth'
    else:
        path = f'./checkpoint/{args.data.dataset}/{args.exp.run}/model_{epoch}.pth'
    return path 

def load_checkpoint(model, epoch=-1):
    path = get_model_path(epoch)
    checkpoint = torch.load(path)
    if args.model.resume:
        wandb.summary['best val acc'] = checkpoint['acc']
        try:
            wandb.summary['best val group acc'] = checkpoint['group_acc']
            wandb.summary['val group acc'] = checkpoint['group_acc']
        except:
            pass
        wandb.summary['epoch'] = checkpoint['epoch']
        if epoch==-1:
            wandb.summary['best val epoch'] = checkpoint['epoch']
    model.module.load_state_dict(checkpoint['net'])
    print(f"...loaded checkpoint from epoch {checkpoint['epoch']} with acc {checkpoint['acc']}")
    return checkpoint['epoch']

def get_clean_val_acc(model):
    """
    Given a directory of model checkpoints, check which one would be used if given a clean validation set. 
    Returns the best validation accuracy and the corresponding test accuracy, as well as the best epoch.
    """
    best_val_acc, best_val_epoch = 0, 0
    for epoch in range(args.exp.num_epochs):
        try:
            checkpoint_epoch = load_checkpoint(model, epoch)
        except:
            raise ValueError(f"Could not load checkpoint for epoch {epoch}")
        old_best_val_acc = best_val_acc
        best_val_acc = train_val_loop(valloader, epoch, phase="val", best_acc=best_val_acc)
        print(f"Old val acc = {old_best_val_acc}, new val acc = {best_val_acc}")
        if best_val_acc > old_best_val_acc:
            best_val_epoch = epoch
        print(f"Epoch {epoch} clean val acc: {best_val_acc}")
    return best_val_acc, best_val_epoch


if (not args.model.resume) and (not args.noise.clean_val):
    best_val_acc, best_test_acc, best_val_epoch = 0, 0, 0
    num_epochs = args.exp.num_epochs if not args.exp.debug else 1
    for epoch in range(num_epochs):
        train_acc = train_val_loop(trainloader, epoch, phase="train")
        best_val_acc = train_val_loop(valloader, epoch, phase="val", best_acc=best_val_acc)
        print(f"Epoch {epoch} val acc: {best_val_acc}")
    if not args.exp.debug:
        best_checkpoint = load_checkpoint(model)
else:
    resume_epoch = args.model.resume_epoch if args.model.resume_epoch != 'best' else -1
    if args.noise.clean_val and args.model.resume_epoch != 'best':
        if not os.path.exists(get_model_path()):
            raise ValueError(f"Could not find checkpoint for {args.exp.run}, clean eval only works for saved checkpoints.")
        resume_epoch = args.model.resume_epoch if args.model.resume_epoch != 'best' else -1
        if args.model.resume_epoch == 'best':
            resume_epoch, best_val_epoch = get_clean_val_acc(model)
            checkpoint_epoch = load_checkpoint(model, epoch=resume_epoch)
        else:
            checkpoint_epoch = load_checkpoint(model, epoch=resume_epoch)

test_acc = train_val_loop(testloader, 0, phase="test")
print(f"Test acc: {test_acc}")