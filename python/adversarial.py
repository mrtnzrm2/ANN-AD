#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:49:39 2020

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
import torch.nn.functional as F
import torch.multiprocessing as mp
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        
        self.lin1 = nn.Linear(1024,16)
        self.lin2 = nn.Linear(16,10)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)
   
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        out = self.dropout2(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        out = self.dropout2(out)
        
        out = self.lin1(out.view(out.size()[0],-1))
        out = self.bn1(out) 
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.lin2(out)
        return out

def plot_nonlinear(model, im, device='cpu', special_name=''):
    im = im.reshape(-1,1,28,28).float().to(device)
    with torch.no_grad():
        fig, ax = plt.subplots(1,im.size(0), figsize=(4,4), dpi=200)
        for i in range(im.size(0)):
            if im.size(0) == 1: ax.imshow(im[i,0,:,:].detach().numpy())
            else: ax[i].imshow(im[i,0,:,:].detach().numpy())
        plt.savefig('../discovery/im_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        model.eval()
        im = model.conv1(im)
        _len_ = im.size(1)
        _num_ = im.size(0)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_cov1_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        
        im = model.bn2(im)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_bn2_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = F.relu(im)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_relu1_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = F.max_pool2d(im, 2, stride=1, padding=0)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_maxpool1_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = model.conv2(im)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_cov2_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = model.bn2(im)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_bn2_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = F.relu(im)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_relu2_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = F.max_pool2d(im, 2, stride=1, padding=0)
        
        fig, ax = plt.subplots(4,4*_num_, figsize=(22,20), dpi=200)
        c = 0
        j = 0
        cc = 0
        for i in range(_num_*_len_):
            if i % 4 == 0 and i != 0:
                j +=1
                c = 0             
            if j < 4: num = 0
            else: num = 1
            if cc%16 == 0: cc = 0
            ax[c,j].imshow(im[num,cc,:,:].detach().numpy())
            c += 1
            cc += 1
        fig.tight_layout(pad=3.0)
        plt.savefig('../discovery/im_maxpool2_{}.png'.format(special_name))
        plt.show()
        plt.close()
        
        im = model.lin1(im.view(im.size(0),-1))
        
        _num_ = im.size(0)
        
        fig, ax = plt.subplots(1,_num_, figsize=(4,4), dpi=200)
        for num in range(_num_):
            if _num_ == 1: ax.imshow(im[num,:].reshape(4,4).detach().numpy())
            else: ax[num].imshow(im[num,:].reshape(4,4).detach().numpy())  
        plt.savefig('../discovery/im_lin1_{}.png'.format(special_name))
        plt.show() 
        plt.close()
        
        
        im = model.bn1(im)
        
        fig, ax = plt.subplots(1,_num_, figsize=(4,4), dpi=200)
        for num in range(_num_):
            if _num_ == 1: ax.imshow(im[num,:].reshape(4,4).detach().numpy())
            else: ax[num].imshow(im[num,:].reshape(4,4).detach().numpy())  
        plt.savefig('../discovery/im_bn1_{}.png'.format(special_name))
        plt.show() 
        plt.close()
        im = F.relu(im)
        fig, ax = plt.subplots(1,_num_, figsize=(4,4), dpi=200)
        for num in range(_num_):
            if _num_ == 1: ax.imshow(im[num,:].reshape(4,4).detach().numpy())
            else: ax[num].imshow(im[num,:].reshape(4,4).detach().numpy())  
        plt.savefig('../discovery/im_relu3_{}.png'.format(special_name))
        plt.show() 
        plt.close()
        # im2 = model.lin2(im)
        fig, ax = plt.subplots(5,2, figsize=(4,10), dpi=200)
        j = 0
        c = 0
        for i in range(10):
            if i%5 == 0 and i != 0:
                j += 1
                c = 0
            ax[c,j].imshow(model.lin2.weight[i,:].reshape(4,4).detach().numpy())
            ax[c,j].axes.get_xaxis().set_visible(False)
            ax[c,j].axes.get_yaxis().set_visible(False)
            c += 1
        plt.savefig('../discovery/im_last_layer_weights_{}.png'.format(special_name))
        plt.show() 
        plt.close()

def pool_gen(queue, event, data, target, model, device, set, max_iter, i):
    if i%100 == 0:
        print('step', i)        
    X = data[i]     
    X = X.to(device)
    
    l = torch.ones((1,), dtype=torch.long)*(target+torch.randint(1,9,(1,)))%10
    l = l.to(device)
    r = l_bfgs(X, l, target, model, set, max_iter=max_iter,device=device)             # CREATE PERTURBATION OF X.
    with torch.no_grad():            
        queue.put((X+r).clamp(min=0))
        event.wait()
        
        if np.random.rand() < 0.01:
            plt.imshow((X+r).clamp(min=0).reshape(28,28))
            plt.show()
            plt.close()

def train(model, 
          train_dataset, 
          test_dataset, 
          check=False,
          device='cpu', 
          set='train', 
          batch_size=150, 
          epsilon=0.001,
          num_epochs=5):
    # Set some hyperparameters
    num_epochs = num_epochs # Number of times to go through training data
    batch_size = batch_size # Batch size to use with training data
    epsilon = epsilon # Learning rate
    test_batch_size = 200 # Batch size to use for test data
    
    # Use cross-entropy loss. 
    # This means we will use the softmax loss function discussed in Notes4*
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD to optimize
    optimizer = torch.optim.AdamW(model.parameters(), lr=epsilon, weight_decay=0.2)
    
    ########
    # There are 60000 images in the training data. If we're using 
    # a batch size of 200, then there are 300 gradient descent steps
    # per epoch. We then repeat this num_epochs times for a total 
    # of num_epochs*steps_per_epoch steps.
    
    if set == 'train':
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=True)
        steps_per_epoch = len(train_loader)
        
        if check:
            check_model(model, train_loader)
    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=test_batch_size,
                                          shuffle=True)
        steps_per_epoch = len(test_loader)
        if check:
            check_model(model, test_loader)
        
        
    total_num_steps = num_epochs*steps_per_epoch
    LossesToPlot=np.zeros(total_num_steps) # Initialize vector of losses
    
    model.train()
    ########
    j=0
    for k in range(num_epochs):
        if set == 'train':
            TrainingIterator=iter(train_loader)
        else:
            TrainingIterator=iter(test_loader)
        for i in range(steps_per_epoch):
    
            # Get one batch of training data, reshape it
            # and send it to the current device        
            X,Y=next(TrainingIterator)
            # X=X.reshape(-1, 28*28)
            X=X.to(device)
            Y=Y.to(device)
          
            # Forward pass: compute yhat and loss for this batch
            Yhat = model(X)
            Loss = criterion(Yhat, Y)
           
            # Backward pass and optimize
            optimizer.zero_grad() # Zero-out gradients from last iteration
            Loss.backward()       # Compute gradients
            optimizer.step()      # Update parameters
            
            # Store loss and increment counter
            LossesToPlot[j]=Loss.item()
            

            j+=1
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(k+1, num_epochs, i+1, steps_per_epoch, Loss.item()))
      
    # Plot losses
    plt.plot(LossesToPlot)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig('loss.png')
    plt.show()
    plt.close()
    
    return model

def create_spicy_dataset(train_dataset, model, device='cpu', train=True, size_train=100, 
                         size_test=100, 
                         download=False,
                         directory='./AMNIST',
                         target=5,
                         parallel=False,
                         max_iter=500):
    #PHASE II: CREATE ADVERSARIAL EXAMPLES, IN THIS CASE 100 ADVERSARIAL EXAMPLES
    #FOR THE NUMBER 5 EXTRACTED FROM MNIST.

    if train:
        
        option_gen = {'size': size_train, 'target':target, 'device': device, 'max_iter': max_iter} 
        if not download:
            if parallel:
                train_adversarial_example, train_adversarial_label = adversarial_gen_mp(train_dataset, model, **option_gen)
            else:
                train_adversarial_example, train_adversarial_label = adversarial_gen(train_dataset, model, **option_gen)
            torch.save(train_adversarial_example, join(directory, 'training_spicy_image_{}.pt'.format(target)))
            torch.save(train_adversarial_label, join(directory, 'training_spicy_label_{}.pt'.format(target)))
        else:
            train_adversarial_example = torch.load(join(directory, 'training_spicy_image_{}.pt'.format(target)))
            train_adversarial_label = torch.load(join(directory, 'training_spicy_label_{}.pt'.format(target)))
        
        #SAVE ADVERSARIAL DATASET TO TRAIN THE NETWORK AGAIN. 
        torch.save([train_adversarial_example,train_adversarial_label], join(directory, 'training_spicy_{}.pt'.format(target)))
        
        with torch.no_grad():
            X = train_adversarial_example.reshape(-1,1,28,28).to(device)
            Y = train_adversarial_label
            Y=Y.to(device)
            Yhat=model(X.float())
            PredictedClass = torch.argmax(Yhat.data, 1)
            TrainingMisclassRate = 1-(PredictedClass == Y).sum().item()/Y.size(0)
            print('Percent of adversarial training_{} images misclassified: {} %'.format(target,100*TrainingMisclassRate))  
    
    else:
        
        option_gen = {'size': size_test, 'target':target, 'device': device, 'max_iter': max_iter} 
        if not download:
            if parallel:  
                test_adversarial_example, test_adversarial_label = adversarial_gen_mp(train_dataset, model, **option_gen)
            else:
                test_adversarial_example, test_adversarial_label = adversarial_gen(train_dataset, model, **option_gen)
            torch.save(test_adversarial_example, join(directory, 'test_spicy_image_{}.pt'.format(target)))
            torch.save(test_adversarial_label, join(directory, 'test_spicy_label_{}.pt'.format(target)))
        else:
            test_adversarial_example = torch.load(join(directory, 'test_spicy_image_{}.pt'.format(target)))
            test_adversarial_label = torch.load(join(directory, 'test_spicy_label_{}.pt'.format(target)))
        
        #SAVE ADVERSARIAL DATASET TO TRAIN THE NETWORK AGAIN. 
        torch.save([test_adversarial_example,test_adversarial_label], join(directory, 'test_spicy_{}.pt'.format(target)))
                                                 
def compute_misclassification(dataset, model, class_amnist=True, print_prediction=False,
                              device='cpu', message='training example'):
       
     with torch.no_grad():
        model.eval()
        if class_amnist == True:
            X = dataset.data.reshape(-1,1,28,28).float().to(device)
            Y = dataset.targets.to(device)
        else:
            X = dataset[0].reshape(-1,1,28,28).float().to(device)
            Y = dataset[1].to(device)
        Yhat=model(X)
        PredictedClass = torch.argmax(Yhat, 1)
        if print_prediction:
            print(PredictedClass)
        TrainingMisclassRate = 1-(PredictedClass == Y).sum().item()/Y.size(0)
        print('Misclassification: {} is {}'.format(message, 100*TrainingMisclassRate)) 
        return 100*TrainingMisclassRate

def join_adversarial_datasets(directory='/Users/admin/Documents/notre_dame/ANN/AMNIST'):

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in files if 'image' not in f and 'label' not in f]
    files = [f for f in files if '.' not in f[0]]
    
    training_files = [training for training in files if 'training' in training and 'spicy_training.pt' not in training]
    test_files = [test for test in files if 'test' in test and 'spicy_test.pt' not in test]
    
    training_torch = [torch.load(directory+'/{}'.format(training)) for  training in training_files]
    test_torch = [torch.load(directory+'/{}'.format(test)) for  test in test_files]
    
    training_size = int(torch.Tensor([training.size(0) for training, _ in training_torch]).sum().item())
    test_size = int(torch.Tensor([test.size(0) for test, _ in test_torch]).sum().item())

    training_img_adversarial_dataset = torch.empty((training_size, 28, 28), dtype=torch.float)
    training_lbl_adversarial_dataset = torch.empty((training_size,))
    test_img_adversarial_dataset = torch.empty((test_size, 28, 28), dtype=torch.float)
    test_lbl_adversarial_dataset = torch.empty((test_size,))
    
    count_a = 0
    count_b = int(training_torch[0][0].size(0))
    for training, labels in training_torch:

        training_img_adversarial_dataset[count_a:count_b,:,:] = training[:,:,:]
        training_lbl_adversarial_dataset[count_a:count_b] = labels[:]
        count_a = count_b
        count_b += int(training.size(0))
    
    count_a = 0
    count_b = int(test_torch[0][0].size(0))
    for test, labels in test_torch:
        test_img_adversarial_dataset[count_a:count_b,:,:] = test[:,:,:]
        test_lbl_adversarial_dataset[count_a:count_b] = labels[:]
        count_a = count_b
        count_b += int(test.size(0))
        
    torch.save([training_img_adversarial_dataset,training_lbl_adversarial_dataset], directory+'/spicy_training.pt')
    torch.save([test_img_adversarial_dataset,test_lbl_adversarial_dataset], directory+'/spicy_test.pt')
    
def check_model(model, data, device='cpu'):
    TrainingIterator=iter(data)
    X,Y=next(TrainingIterator)
    X=X.to(device)
    print('Output size = ',model(X).size())

def print_dataset(dataset, class_amnist=True):
    if class_amnist == True:
        for i, im in enumerate(dataset.data):
            plt.imshow(im[:,:].cpu())
            plt.show()
            plt.close()
    else:
        for i, im in enumerate(dataset[0]):
            plt.imshow(im.cpu())
            plt.show()
            plt.close()  
            
def adversarial_gen(train_dataset, model, device='cpu', size= 100, target=5,
                    set=1, max_iter=500):
      
    dim = train_dataset.data.size()[-1] 
    adversarial_ex = torch.zeros([size, dim, dim], dtype=torch.float)
    Data, Targets = train_dataset.data, train_dataset.targets
    Data = Data[ Targets==target ]    
    Data = Data[torch.randint(0, Data.size(0), (Data.size(0),))]
    
    for i in np.arange(0,len(Data),set):
        if i >= size:
            break
        if i%100 == 0:
            print('step', i)        
        X = Data[i]     
        X = X.to(device)
        
        l = torch.ones((1,), dtype=torch.long)*(target+torch.randint(1,9,(1,)))%10
        l = l.to(device)
        r = l_bfgs(X, l, target, model, set, max_iter=max_iter, device=device)             # CREATE PERTURBATION OF X.

        with torch.no_grad():
            adversarial_ex[i,:, :] =  (X+r).clamp(min=0)             
            if np.random.rand() < 0.001:
                plt.imshow(adversarial_ex[i,:,:])
                plt.show()
                plt.close()
    return (adversarial_ex.reshape(-1,28,28), torch.ones((size,))*target)

def adversarial_gen_mp(train_dataset, model, device='cuda', size= 100, target=5,
                    set=1, max_iter=500):
    dim = train_dataset.data.size()[-1] 
    Data, Targets = train_dataset.data, train_dataset.targets
    Data = Data[ Targets==target ]    
    Data = Data[torch.randint(0, Data.size(0), (Data.size(0),))]
    
    example = torch.empty((size,dim,dim), dtype=torch.float).to(device)
    mp.set_sharing_strategy('file_system') 
    done = mp.Event()
    processes = []
    done_queue = mp.Queue()
    
    for i in np.arange(0, size,set):
        p = mp.Process(target=pool_gen, args=(done_queue, done, Data, target, model, device, set, max_iter, i))
        processes.append(p)
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    for h in range(size):
        example[h,:,:] = done_queue.get() 
    done.set()
    # plt.imshow(example[0,:,:].reshape(28,28))  
    # plt.show()
    # plt.close()          
    return (example.reshape(-1,28,28), torch.ones((size,))*target)

def l_bfgs(X, l, target, model, set, device='cpu', max_iter = 1):
    epsilon = 0.001 # Learning rate      
    criterion = nn.CrossEntropyLoss()
    r_1 = torch.rand(1, 28, 28, device=device).float().requires_grad_(True)   #INITIAL GUESS
    optimizer = torch.optim.LBFGS([r_1], lr=epsilon, max_iter=max_iter,
                              history_size=400, tolerance_change=1e-12,
                              tolerance_grad=1e-11)

    def closure():
        
        optimizer.zero_grad()
        model.eval() 
        _X_ = (X+r_1).clamp(min=0)
        Yhat = model(_X_.reshape(-1,1,28,28).float())
        loss = criterion(Yhat, l) + 0.2*torch.sum(torch.norm(r_1, p=2, dim=1))
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    with torch.no_grad():           
        print('Miss targets', l)
        model.eval()
        _X_= (X+r_1).clamp(min=0)
        Yhat = model(_X_.reshape(-1,1,28,28).float())
        PredictedClass = torch.argmax(Yhat,1)
        print('Prediction', PredictedClass)
        # plot_nonlinear(model, _X_, device=device, special_name='{}_{}'.format(PredictedClass[0].item(), l[0].item()))
        if np.random.rand() < 1:
                plt.imshow((r_1).reshape(28,28).cpu().detach().numpy(), cmap='gray')
                plt.show()
                plt.close()
                plt.imshow(X.reshape(28,28).cpu(), cmap='gray')
                plt.show()
                plt.close()
                plt.imshow(_X_.reshape(28,28).cpu(), cmap='gray')
                plt.show()
                plt.close()

    return r_1

if __name__ == '__main__':
    directory = join(os.getcwd(), '..')
    #mp.set_start_method('spawn', force=True)
    #mp.set_start_method('spawn')
    #PHASE I: CREATE MODEL FROM MNIST TRAINING DATASET
    train_dataset = MNIST(root=directory, 
                          train=True, 
                          transform=transforms.ToTensor(),  
                          download=False)
    test_dataset = MNIST(root=directory, 
                          train=False,
                          transform=transforms.ToTensor(),  
                          download=False)
    

