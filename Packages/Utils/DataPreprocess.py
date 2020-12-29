#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import preprocessing
import pandas as pd
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_window(dataframe, data, data_gt, win_size, step_size, start_col, end_col, gt_start_col, gt_end_col, axis_no):
    len_df = dataframe.shape[0]
    narray = dataframe.values
    for i in range(0, len_df, step_size):
        window = narray[i:i+win_size, start_col:end_col]

        if window.shape[0] != win_size:
            continue
        else:
            reshaped_window = window.reshape(1,win_size,1,axis_no)
            gt = np.bincount(narray[i:i+win_size,gt_start_col:gt_end_col].astype(int).ravel()).argmax()
            
            data.append(reshaped_window)
            data_gt.append(gt)

def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]



def valid_loader(valid_x, valid_y, batch_size=64):

    valid_x_swap = np.swapaxes(valid_x,1,3)
    valid_set = data_loader(valid_x_swap, valid_y)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return valid_loader


class data_loader(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)
    
    
# Used in DANN Baseline 
def load(train_x, train_y, test_x, test_y, batch_size=32):
  
    train_x_swap = np.swapaxes(train_x,1,3)
    test_x_swap = np.swapaxes(test_x,1,3)

    train_set = data_loader(train_x_swap, train_y)
    test_set = data_loader(test_x_swap, test_y)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader

def test_load(test_x, test_y, batch_size=64):

    test_x_swap = np.swapaxes(test_x,1,3)
    test_set = data_loader(test_x_swap, test_y)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return test_loader


def load_train_valid_test(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=64):
  
    train_x_swap = np.swapaxes(train_x,1,3)
    valid_x_swap = np.swapaxes(valid_x,1,3)
    test_x_swap = np.swapaxes(test_x,1,3)

    train_set = data_loader(train_x_swap, train_y)
    valid_set = data_loader(valid_x_swap, valid_y)
    test_set = data_loader(test_x_swap, test_y)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader
    

def dctn_load(source, ground_truth, batch_size=64):
    source = np.swapaxes(source,1,3)
    dataset = data_loader(source, ground_truth)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def combined_dataset_load(source, ground_truth, batch_size=64):
#     source = np.swapaxes(source,1,3)
    dataset = data_loader(source, ground_truth)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


def standardization_and_normalization(dataframe):
    scaler = preprocessing.StandardScaler()
    df_standardized = scaler.fit_transform(dataframe)
    return df_standardized

#     df_standardized = pd.DataFrame(df_standardized)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     np_scaled = min_max_scaler.fit_transform(df_standardized)
#     return np_scaled

#======================== Data Augmentatio =========================================

def DA_Jitter(X, sigma=0.01):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis,angle))

sigma = 0.005
knot = 2

## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph

    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])

    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

class modified_data_loader(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = np.squeeze(self.samples[index], axis = 1)  # (3,1,128) -> (3,128)
        x = np.swapaxes(x, 0, 1) # (128,3)
        
        modified_signal = DA_Jitter(DA_Scaling(DA_TimeWarp(DA_Rotation(x), sigma = 0.2)))      
        modified_signal = modified_signal.reshape(128,1,3)
        modified_signal = np.swapaxes(modified_signal, 0, 2) #(1,3,1,128)
          
        return modified_signal, self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)
    
    
def modified_load_train_valid_test(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=64):
    train_x_swap = np.swapaxes(train_x,1,3)
    valid_x_swap = np.swapaxes(valid_x,1,3)
    test_x_swap = np.swapaxes(test_x,1,3)
    
    train_set = modified_data_loader(train_x_swap, train_y)
    valid_set = modified_data_loader(valid_x_swap, valid_y)
    test_set = modified_data_loader(test_x_swap, test_y)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader


# DANN Training and Testing
def dann_train(model, optimizer, dataloader_src, dataloader_tar, source_test_loader, target_test_loader, N_EPOCH, DEVICE, BATCH_SIZE, no_epochs_to_average_accuracy):
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    
    acc_sum = 0
    for epoch in range(1, N_EPOCH+1):
        model.train()
        len_dataloader = min(len(dataloader_src), len(dataloader_tar))
        data_src_iter = iter(dataloader_src)
        data_tar_iter = iter(dataloader_tar)

        i = 1
        print("Epoch: "+str(epoch)+ " Train loader: "+ str(len_dataloader))
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / N_EPOCH / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # Training model using source data
            data_source = data_src_iter.next()
            optimizer.zero_grad()
            
            s_img, s_label = data_source[0].to(DEVICE), data_source[1].to(DEVICE).long()
            domain_label = torch.zeros(BATCH_SIZE).long().to(DEVICE)
            class_output, domain_output = model(input_data=s_img, alpha=alpha)
            
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            data_target = data_tar_iter.next()
            t_img = data_target[0].to(DEVICE)
            domain_label = torch.ones(BATCH_SIZE).long().to(DEVICE)
            _, domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_t_domain + err_s_domain

            err.backward()
            optimizer.step()

            i += 1

        acc_src, src_f1_micro, src_precision_micro, src_recall_micro = dann_test(model, source_test_loader, "Source", epoch, N_EPOCH, DEVICE)
        acc_tar, tar_f1_micro, tar_precision_micro, tar_recall_micro = dann_test(model, target_test_loader, "Target", epoch, N_EPOCH, DEVICE)
        
        if ((N_EPOCH - epoch >= 0) & (N_EPOCH - epoch <= no_epochs_to_average_accuracy)):
            acc_sum += acc_tar
    return acc_sum/no_epochs_to_average_accuracy, tar_f1_micro, tar_precision_micro, tar_recall_micro
    
    
    
def dann_test(model, dataloader, dataset_name, epoch, N_EPOCH, DEVICE):
    alpha = 0
    model.eval()
    n_correct = 0
    
    prediction = []
    ground_truth = []
    
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE).long()
            class_output, _ = model(input_data=t_img, alpha=alpha)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()
            
            prediction.extend(pred[1].tolist())
            ground_truth.extend(t_label.cpu().numpy().tolist())

    accu = float(n_correct) / len(dataloader.dataset) * 100
    f1_micro        = f1_score(ground_truth,        prediction, average='micro')
    precision_micro = precision_score(ground_truth, prediction, average='micro')
    recall_micro    = recall_score(ground_truth,    prediction, average='micro')
    
    print('Epoch: [{}/{}], accuracy on {} dataset: {:.4f}%'.format(epoch, N_EPOCH, dataset_name, accu))
    return accu, f1_micro, precision_micro, recall_micro



# import zipfile as zf
# files = zf.ZipFile("/notebooks/Downloads/ISWC/Baselines/DANN.zip", 'r')
# files.extractall('/notebooks/Downloads/ISWC/Baselines/')
# files.close()


# class ActivityDataset(Dataset):
#     def __init__(self, low, high):
#         self.samples = list(range(low, high))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
        
#         n = self.samples[idx]
#         successors = torch.arange(4).float() + n + 1
#         noisy = torch.randn(4) + successors
#         return n, successors, noisy

# class modified_data_loader(Dataset):
#     def __init__(self, samples, labels, transform=None):
#         self.samples = samples
#         self.labels = labels
#         self.transform = transform
#         print(str("Called Sample shape: " + self.samples.shape))

#     def __getitem__(self, index):
#         return self.samples[index], self.labels[index]

#     def __len__(self):
#         return len(self.samples)
    
    
# def modified_load_train_valid_test(train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=64):
#     train_x_swap = np.swapaxes(train_x,1,3)
#     valid_x_swap = np.swapaxes(valid_x,1,3)
#     test_x_swap = np.swapaxes(test_x,1,3)
    
#     print("Shape: "+ str(train_x_swap.shape))

#     train_set = modified_data_loader(train_x_swap, train_y)
#     valid_set = modified_data_loader(valid_x_swap, valid_y)
#     test_set = modified_data_loader(test_x_swap, test_y)
    
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
#     return train_loader, valid_loader, test_loader
