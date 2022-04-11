#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:42:16 2020

@author: jorgem
"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
    
class AMNIST():
    _repr_indent = 4

    def __init__(self, root, training_file='spicy_training.pt', test_file='spicy_test.pt',
                 train=True, transform=None, target_transform=None):

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.training_file = training_file
        self.test_file = test_file
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
            
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root)
    
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)
    
    def extra_repr(self):
        return ""
    
if __name__ == '__main__':
    spicy_train_dataset = AMNIST('./AMNIST', transform=transforms.ToTensor())
