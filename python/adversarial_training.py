#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:51:35 2020

@author: jorgem
"""

import torch # We no longer import as tch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms # Contains MNIST, other image datasets, and image processing functions
import matplotlib.pyplot as plt
import os
from os.path import join
from torchvision.datasets import MNIST # Contains data sets and functions for image processing
from amnist import AMNIST
from adversarial_examples import MyMNISTclassifier, train, compute_misclassification

def merge_datasets(dat1, dat2):
    img1, label1 = dat1
    img2, label2 = dat2
    
    img3 = torch.empty((img1.size(0)+img2.size(0), 28, 28), dtype=torch.uint8)
    label3 = torch.empty((img1.size(0)+img2.size(0),))
    
    img3[:img1.size(0),:,:] = img1[:,:,:]
    img3[img1.size(0):,:,:] = img2[:,:,:]
    
    label3[:img1.size(0)] = label1[:]
    label3[img1.size(0):] = label2[:]
    
    return (img3, label3)

def split_dataset(dataset, fragments):
    
    img, label = dataset.data, dataset.targets
    fragmentation = []
    
    for fra in fragments:
        torch_fragment_img = torch.empty((fra, 28, 28), dtype=torch.uint8)
        torch_fragment_lab = torch.empty((fra,))
        
        idx = torch.randint(0, img.size(0), (fra,))
        torch_fragment_img[:,:,:] = img[idx, :, :]
        torch_fragment_lab[:] = label[idx]
        
        fragmentation.append((torch_fragment_img, torch_fragment_lab))
    return fragmentation

def save_dataset(dataset, name, directory):
    torch.save(dataset, join(directory, name))
    
def save_mix_dataset(dataset, train_dataset, directory, pieces, trainig=True):
    '''
    Pieces must be an array or list of integers. Example of implementation:
        
    pieces = np.linspace(10000,50000,10, dtype=int)
    save_mix_dataset(adversarial_train_dataset, train_dataset, directory, pieces)
    '''
    preprocessed_dataset = (dataset.data, dataset.targets)    
    fragments = split_dataset(train_dataset, pieces)
    
    for i, fra in enumerate(fragments):
       fra_img, fra_label = merge_datasets(fra, preprocessed_dataset)
       if trainig:
           save_dataset((fra_img, fra_label), 'mix_dataset_{}.pt'.format(pieces[i]), join(directory,'mix_dataset'))
       else:
           save_dataset((fra_img, fra_label), 'mix_dataset_{}_test.pt'.format(pieces[i]), join(directory,'mix_dataset'))
    
if __name__ == '__main__':
    directory = '/Users/admin/Documents/ND/ANN'
      #PHASE I: CREATE MODEL FROM MNIST TRAINING DATASET
    train_dataset = MNIST(root=directory, 
                          train=True, 
                          transform=transforms.ToTensor(),  
                          download=False)
    test_dataset = MNIST(root=directory, 
                          train=False,
                          transform=transforms.ToTensor(),  
                          download=False)
    
    print('---Train the neural network.---')
    train_model = False

    if train_model:
        model = train(MyMNISTclassifier(), train_dataset, test_dataset)
        torch.save(model.state_dict(), join(directory, 'trained_neural_networks/model_dic.pt'))
        
    else:
        model = MyMNISTclassifier()
        model.load_state_dict(torch.load(join(directory, 'trained_neural_networks/model_dic.pt')))
        model.eval()
    print('---Training model with adversarial + x training samples.---')
    #PHASE III: TRAIN THE PREVIOUS ANN WITH ADVERSARIAL EXAMPLES
    
    adversarial_train_dataset = AMNIST(join(directory, 'AMNIST'), train=True, transform=transforms.ToTensor())
    adversarial_test_dataset = AMNIST(join(directory, 'AMNIST'), train=False, transform=transforms.ToTensor())
    
    pieces = np.linspace(10000, 50000, 10, dtype=int)
    save_mix_dataset(adversarial_train_dataset, train_dataset, directory, pieces)
    
    
    missclassification = np.zeros((10,))
    i = 0
    
    for pie in pieces:
        for file in os.listdir(join(directory,'mix_dataset')):
            if '.' == file[0]: continue
            if str(pie) in file:
                model_training_dataset = AMNIST(join(directory,'mix_dataset'), training_file=file,
                                        train=True, transform=transforms.ToTensor())
                model = train(MyMNISTclassifier(), model_training_dataset, 0)
                
                x = compute_misclassification(test_dataset, model, message=file[:-3]+' with respect to test_dataset')     
                missclassification[i] = x
                i += 1
                
    plt.figure(figsize=(10,8) ,dpi=200)
    ax = plt.gca()
    ax.plot(pieces, missclassification, '--')
    ax.set_xlabel('10000 adversarial + x training')
    ax.set_ylabel('Missclassifaction rate')
    plt.savefig(join(directory,'figures'))
    plt.show()
    plt.close()
    
    
    compute_misclassification(adversarial_train_dataset, model, message='adversarial training dataset')
    compute_misclassification(train_dataset, model, message='training dataset after retraining')
    compute_misclassification(adversarial_test_dataset, model, message='adversarial test dataset')