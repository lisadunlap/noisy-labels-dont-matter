import os
import sys
import time
import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import ast
from sklearn.metrics import confusion_matrix
from scipy import stats

import omegaconf

def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            #print(running_key_temp, value)
            flattened_dict[running_key_temp] = value
    return flattened_dict

def read_unknowns(unknown_list):
    """
    input is of form ['--METHOD.MODEL.LR=0.001758722642964502', '--METHOD.MODEL.NUM_LAYERS=1']
    """
    ret = {}
    for item in unknown_list:
        key, value = item.split('=')
        try:
            value = ast.literal_eval(value)
        except:
            print("MALFORMED ", value)
        k = key[2:]
        ret[k] = value
    return ret

def nest_dict(flat_dict, sep='.'):
    """Return nested dict by splitting the keys on a delimiter.

    >>> from pprint import pprint
    >>> pprint(nest_dict({'title': 'foo', 'author_name': 'stretch',
    ... 'author_zipcode': '06901'}))
    {'author': {'name': 'stretch', 'zipcode': '06901'}, 'title': 'foo'}
    """
    tree = {}
    for key, val in flat_dict.items():
        t = tree
        prev = None
        for part in key.split(sep):
            if prev is not None:
                t = t.setdefault(prev, {})
            prev = part
        else:
            t.setdefault(prev, val)
    return tree

def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

def evaluate(predictions, labels, groups=[], label_names=None, num_augmentations=1):
    """
    Gets the evaluation metrics given the predictions and labels. 
    num_augmentations is for test-time augmentation, if its set >1, we group predictions by 
    num_augmentations and take the consesus as the label
    """
    if num_augmentations > 1:
        predictions_aug = predictions.reshape((int(len(predictions)/num_augmentations), num_augmentations))
        print("aug shape ", predictions_aug.shape, " label shape ", labels.shape)
        majority_pred = []
        for i, group in enumerate(predictions_aug):
            majority_pred.append(stats.mode(group)[0])
        predictions = np.array(majority_pred)
        print("new pred shape ", predictions.shape)

    cf_matrix = confusion_matrix(labels, predictions, labels=label_names)
    class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
    accuracy = np.mean((labels == predictions).astype(np.float)) * 100.
    balanced_acc = class_accuracy.mean()
    if len(groups) == 0:
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy])
    else:
        group_acc = np.array([get_per_group_acc(value, predictions, labels, groups) for value in np.unique(groups)])
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy]), np.array([round(g,2) for g in group_acc])

def get_per_group_acc(value, predictions, labels, groups):
    indices = np.array(np.where(groups == value))
    return np.mean((labels[indices] == predictions[indices]).astype(np.float)) * 100.
