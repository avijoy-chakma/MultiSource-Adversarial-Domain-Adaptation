#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://stackoverflow.com/questions/6723426/looping-over-arrays-printing-both-index-and-value


# In[1]:


home_url = "/notebooks"


# In[2]:

import sys
sys.path.append(home_url+'/Downloads/ISWC/Debug-Packages/Utils/')
sys.path.append(home_url+'/Downloads/ISWC/Debug-Packages/Model/')
from msda_classifier import *
from msda_extractor import *
from msda_discriminator import *
from DataPreprocess import *
from Util import *
from Visualization import *
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import math
import csv
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import tqdm
from pylab import rcParams
import itertools
import seaborn as sns
import time
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
np.set_printoptions(precision=2)
torch.manual_seed(10)
np.random.seed(10)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--win_size")
parser.add_argument("--overlap")
parser.add_argument("--activity")

parser.add_argument("--source1")
parser.add_argument("--source2")
parser.add_argument("--target")
parser.add_argument("--position")



args = parser.parse_args()

win_size=int(args.win_size)
overlap = float(args.overlap)
activity_num = int(args.activity)
position = int(args.position)
source1_user = int(args.source1)
source2_user = int(args.source2)
target_user = int(args.target)


step_size=int(win_size*overlap)
gpu_id= 0
activities = [ "standing", "lying-back", "ascending", "walking-parking-lot", "treadmill-running", "stepper-exercise", "cross-trainer-exercise", "rowing", "jumping",  "playing-basketball"]
data_source = {"source1":source1_user, "source2": source2_user, "target":target_user}

print(data_source)

# In[5]:

item = ["train","valid","test"]
person_list = ["User1","User2","User3","User4", "User5","User6","User7","User8"]
# position_array = ["TORSO","RA","LA","RL","LL"]
position_array = ["LA","RL","LL"]

train_percentage = 0.6
valid_percentage = 0.2


AXIS = 3
FROM = 0
TO = FROM+3
START = 4
END = 5

N_EPOCH = 100
batch_size = 32
LEARNING_RATE = 0.0001
beta1=0.9
beta2=0.99

plot_common_title = " window: "+ str(win_size) + "--Overlap: "+ str(overlap) + "--Position: "+ position_array[position] + "\n"+ "--Activity: "+ str(activity_num)


# In[10]:


folder_name = str(activity_num)+ " Activity"+"_Window "+str(win_size)+ "_Overlap "+str(overlap)
user_combination = "/S1: "+str(source1_user)+" S2: "+str(source2_user)+" Target "+ str(target_user)+"/"
save_path = "/Pre-trained/DSADS-SamePosition-DifferentUser-augmented-10/"+folder_name+"/Position-"+position_array[position]+user_combination

dataset_path = home_url+"/Downloads/ISWC/Preprocessing/DSADS/Data Files/"+folder_name+"/"
classifier_save_path = os.getcwd() +save_path
plot_save_path = classifier_save_path
    
if not os.path.exists(classifier_save_path):
    os.makedirs(classifier_save_path)
    
if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)


# In[11]:

experiment_summery_file = os.getcwd() + "/Pre-trained/DSADS-SamePosition-DifferentUser-augmented-10/"
file1 = open(experiment_summery_file+"DSADS-SamePosition-DifferentUser-augmented-10.txt", "a")

feature_save_path = classifier_save_path+ "extractor.pth"
classifier_A_save_path = classifier_save_path + "classifierA.pth"
classifier_B_save_path = classifier_save_path + "classifierB.pth"


# In[12]:


# print("Classifier Saving Path: "+ classifier_A_save_path)
# print("\nPlot Saving Path: "+ plot_save_path)
# print("\nDataset Path: "+ dataset_path)


# In[13]:


DEVICE = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')
A_result = []
B_result = []


# - Data Structure

# In[14]:


s1_train = []
s2_train = []
t_train = []

s1_gt_train = []
s2_gt_train = []
t_gt_train = []

s1_valid = []
s2_valid = []
t_valid = []

s1_gt_valid = []
s2_gt_valid = []
t_gt_valid = []

s1_test = []
s2_test = []
t_test = []

s1_gt_test = []
s2_gt_test = []
t_gt_test = []


# # Load Data Files and Preprocess

# In[17]:



for key in data_source:
    for split_index in range(0,3):

        if key == "source1":
            file_name = person_list[source1_user-1] + "_" + position_array[position]+'_'+item[split_index]
#             print(file_name)
            df = pd.read_csv(dataset_path+file_name+'.csv', sep=",")  

            if split_index == 0:
                calculate_window(df, s1_train, s1_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, s1_valid, s1_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, s1_test, s1_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
        elif key == "source2":
            file_name = person_list[source2_user-1] + "_" + position_array[position]+'_'+item[split_index]
#             print(file_name)
            df = pd.read_csv(dataset_path+file_name+'.csv', sep=",") 

            if split_index == 0:
                calculate_window(df, s2_train, s2_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, s2_valid, s2_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, s2_test, s2_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
        elif key == "target":
            file_name = person_list[target_user-1] + "_" + position_array[position]+'_'+item[split_index]
#             print(file_name)
            df = pd.read_csv(dataset_path+file_name+'.csv', sep=",") 

            if split_index == 0:
                calculate_window(df, t_train, t_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, t_valid, t_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, t_test, t_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)


# - Source1 Data Processing

# In[18]:


s1_train = np.concatenate( s1_train, axis=0 ).astype(np.float32)
s1_gt_train = np.array(s1_gt_train  ).astype(np.float32)
s1_valid = np.concatenate( s1_valid, axis=0 ).astype(np.float32)
s1_gt_valid = np.array(s1_gt_valid  ).astype(np.float32)
s1_test = np.concatenate( s1_test, axis=0 ).astype(np.float32)
s1_gt_test = np.array(s1_gt_test  ).astype(np.float32)


# - Source2 Data Processing

# In[19]:


s2_train = np.concatenate( s2_train, axis=0 ).astype(np.float32)
s2_gt_train = np.array(s2_gt_train  ).astype(np.float32)
s2_valid = np.concatenate( s2_valid, axis=0 ).astype(np.float32)
s2_gt_valid = np.array(s2_gt_valid  ).astype(np.float32)
s2_test = np.concatenate( s2_test, axis=0 ).astype(np.float32)
s2_gt_test = np.array(s2_gt_test  ).astype(np.float32)


# - Target Data Processing

# In[20]:


t_train = np.concatenate( t_train, axis=0 ).astype(np.float32)
t_gt_train = np.array(t_gt_train  ).astype(np.float32)
t_valid = np.concatenate( t_valid, axis=0 ).astype(np.float32)
t_gt_valid = np.array(t_gt_valid  ).astype(np.float32)
t_test = np.concatenate( t_test, axis=0 ).astype(np.float32)
t_gt_test = np.array(t_gt_test  ).astype(np.float32)


# In[21]:


# s1_train.shape


# In[22]:


plot_training_data_distribution_multi_source(s1_gt_train, s2_gt_train, t_gt_train, activities, plot_save_path)


# In[23]:


output_gt_number = len(np.unique(s1_gt_train))


# # Model Training

# In[24]:



def train(extractor, classifier_A, classifier_B, optimizer, S1_train, S1_valid, S1_test, S2_train, S2_valid, S2_test):
    n_batch = len(S1_train.dataset) // batch_size
    criterion = nn.CrossEntropyLoss()

    for e in range(N_EPOCH):
        
        extractor.train()
        classifier_A.train()
        classifier_B.train()
        
        B_train_itr, B_valid_itr, B_test_itr = iter(S2_train), iter(S2_valid), iter(S2_test)
        
        A_correct, B_correct = 0, 0
        A_total, B_total = 0, 0
        total_loss = 0

        A_train_loss, A_valid_loss = 0, 0
        B_train_loss, B_valid_loss = 0, 0
        
        for index, (Modified_A_Sample, A_sample, A_target) in enumerate(S1_train):
            
            try:
                Modified_B_Sample, B_sample, B_target = B_train_itr.next()
            except StopIteration:
                B_train_itr = iter(S2_train)
                Modified_B_Sample, B_sample, B_target = B_train_itr.next()

            
            Modified_A_Sample, A_sample, A_target = Modified_A_Sample.to(DEVICE).float(), A_sample.to(DEVICE).float(), A_target.to(DEVICE).long()        
            A_sample = A_sample.view(-1, AXIS, 1, win_size)
            Modified_A_Sample = Modified_A_Sample.view(-1, AXIS, 1, win_size)
            
            Modified_B_Sample, B_sample, B_target = Modified_B_Sample.to(DEVICE).float(), B_sample.to(DEVICE).float(), B_target.to(DEVICE).long()        
            B_sample = B_sample.view(-1, AXIS, 1, win_size)
            Modified_B_Sample = Modified_B_Sample.view(-1, AXIS, 1, win_size)
            

            A_feature = extractor(A_sample)
            A_output = classifier_A(A_feature)
            
            Modified_A_feature = extractor(Modified_A_Sample)
            Modified_A_output = classifier_A(Modified_A_feature)
            
            B_feature = extractor(B_sample)
            B_output = classifier_B(B_feature)
            
            Modified_B_feature = extractor(Modified_B_Sample)
            Modified_B_output = classifier_B(Modified_B_feature)
            
            
            A_loss = criterion(A_output, A_target)
            B_loss = criterion(B_output, B_target)
            
            Modified_A_loss = criterion(Modified_A_output, A_target)
            Modified_B_loss = criterion(Modified_B_output, B_target)

            A_train_loss += A_loss
            B_train_loss += B_loss
            
            total_loss = sum([A_loss, B_loss, Modified_A_loss, Modified_B_loss])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss += total_loss.item()
            
            _, A_predicted = torch.max(A_output.data, 1)
            A_total += A_target.size(0)
            A_correct += (A_predicted == A_target).sum()
            
            _, B_predicted = torch.max(B_output.data, 1)
            B_total += B_target.size(0)
            B_correct += (B_predicted == B_target).sum()

#             if index % 20 == 0:
#                 tqdm.tqdm.write('Epoch: [{}/{}], Batch: [{}/{}], A_loss:{:.4f}, B_loss:{:.4f}'.format(e + 1,
#                             N_EPOCH, index + 1, n_batch,A_loss.item(), B_loss.item()))
            
            
        A_acc_train = float(A_correct) * 100/ A_total
        B_acc_train = float(B_correct) * 100/ B_total

#         tqdm.tqdm.write('Epoch: [{}/{}], A Train Acc: {:.2f}%, B acc: {:.2f}%'.format(e + 1, N_EPOCH, A_acc_train, B_acc_train))

        # Validating
        extractor.eval()
        classifier_A.eval()
        classifier_B.eval()
        
        a_prediction = []
        b_prediction = []
        a_ground_truth = []
        b_ground_truth = []
        
        with torch.no_grad():
            A_correct, B_correct = 0, 0
            A_total, B_total = 0, 0
            correct, total = 0, 0
            
#             for valid_index, (A_sample, A_target) in enumerate(S1_valid):
            for valid_index, (Modified_A_sample, A_sample, A_target) in enumerate(S1_valid):
                try:
                    Modified_B_sample, B_sample, B_target = B_valid_itr.next()
                except StopIteration:
                    B_valid_itr = iter(S2_valid)
                    Modified_B_sample, B_sample, B_target = B_valid_itr.next()
                    
                
                Modified_A_Sample, A_sample, A_target = Modified_A_Sample.to(DEVICE).float(), A_sample.to(DEVICE).float(), A_target.to(DEVICE).long()        
                A_sample = A_sample.view(-1, AXIS, 1, win_size)
                Modified_A_Sample = Modified_A_Sample.view(-1, AXIS, 1, win_size)

                Modified_B_Sample, B_sample, B_target = Modified_B_Sample.to(DEVICE).float(), B_sample.to(DEVICE).float(), B_target.to(DEVICE).long()        
                B_sample = B_sample.view(-1, AXIS, 1, win_size)
                Modified_B_Sample = Modified_B_Sample.view(-1, AXIS, 1, win_size)

                
                feature = extractor(A_sample)    
                A_output = classifier_A(feature)
                
                feature = extractor(B_sample)
                B_output = classifier_B(feature)
                
                _, A_predicted = torch.max(A_output.data, 1)
                A_total += A_target.size(0)
                A_correct += (A_predicted == A_target).sum()
                
                _, B_predicted = torch.max(B_output.data, 1)
                B_total += B_target.size(0)
                B_correct += (B_predicted == B_target).sum()
                
                # For loss calculation purpose only
                A_valid_loss = criterion(A_output, A_target)
                B_valid_loss = criterion(B_output, B_target)

                A_valid_loss += A_loss
                B_valid_loss += B_loss
                
                # Save the intermediate results for intermittent confusion matrix plot - Debug Purpose
#                 a_prediction.extend(A_predicted.tolist())
#                 b_prediction.extend(B_predicted.tolist())
#                 a_ground_truth.extend(A_target.cpu().numpy().tolist())
#                 b_ground_truth.extend(B_target.cpu().numpy().tolist())
    
        A_acc_valid = float(A_correct) * 100/ A_total
        B_acc_valid = float(B_correct) * 100/ B_total
#         tqdm.tqdm.write('Epoch: [{}/{}], A Valid Acc: {:.2f}%, B Valid Acc: {:.2f}%'.format(e + 1, N_EPOCH, A_acc_valid, B_acc_valid))
        
        
        
        ## Testing on the source domain test split - For overfitting test
        A_correct, B_correct = 0, 0
        A_total, B_total = 0, 0       
        correct, total = 0, 0
            
        for test_index, (Modified_A_sample, A_sample, A_target) in enumerate(S1_test):
            try:
                Modified_B_sample, B_sample, B_target = B_test_itr.next()
            except StopIteration:
                B_test_itr = iter(S2_test)
                Modified_B_sample, B_sample, B_target = B_test_itr.next()


#             A_sample, A_target = A_sample.to(DEVICE).float(), A_target.to(DEVICE).long()
#             A_sample = A_sample.view(-1, AXIS, 1, win_size)

#             B_sample, B_target = B_sample.to(DEVICE).float(), B_target.to(DEVICE).long()
#             B_sample = B_sample.view(-1, AXIS, 1, win_size)
            
            Modified_A_Sample, A_sample, A_target = Modified_A_Sample.to(DEVICE).float(), A_sample.to(DEVICE).float(), A_target.to(DEVICE).long()        
            A_sample = A_sample.view(-1, AXIS, 1, win_size)
            Modified_A_Sample = Modified_A_Sample.view(-1, AXIS, 1, win_size)
            
            Modified_B_Sample, B_sample, B_target = Modified_B_Sample.to(DEVICE).float(), B_sample.to(DEVICE).float(), B_target.to(DEVICE).long()        
            B_sample = B_sample.view(-1, AXIS, 1, win_size)
            Modified_B_Sample = Modified_B_Sample.view(-1, AXIS, 1, win_size)

            feature = extractor(A_sample)    
            A_output = classifier_A(feature)

            feature = extractor(B_sample)
            B_output = classifier_B(feature)

            _, A_predicted = torch.max(A_output.data, 1)
            A_total += A_target.size(0)
            A_correct += (A_predicted == A_target).sum()

            _, B_predicted = torch.max(B_output.data, 1)
            B_total += B_target.size(0)
            B_correct += (B_predicted == B_target).sum()


        A_acc_test = float(A_correct) * 100 / A_total
        B_acc_test = float(B_correct) * 100 / B_total
        
        
        A_result.append([A_acc_train, A_acc_valid, A_acc_test, A_train_loss, A_valid_loss])
        B_result.append([B_acc_train, B_acc_valid, B_acc_test, B_train_loss, B_valid_loss])
        
        A_result_np = np.array(A_result, dtype=float)
        B_result_np = np.array(B_result, dtype=float)
        
        np.savetxt(plot_save_path+"A_result.csv", A_result_np, fmt='%.2f', delimiter=',')
        np.savetxt(plot_save_path+"B_result.csv", B_result_np, fmt='%.2f', delimiter=',')
        
        torch.save(extractor.state_dict(),feature_save_path)
        torch.save(classifier_A.state_dict(),classifier_A_save_path)
        torch.save(classifier_B.state_dict(),classifier_B_save_path)
        
    evaluation(e)


# # Target Domain Test Dataset Evaluation

# In[26]:


def evaluation(e):
    extractor = AccExtractor().to(DEVICE)
    classifier_A = AccClassifier(output_gt_number).to(DEVICE)
    classifier_B = AccClassifier(output_gt_number).to(DEVICE)

    extractor.load_state_dict(torch.load(feature_save_path))
    classifier_A.load_state_dict(torch.load(classifier_A_save_path))
    classifier_B.load_state_dict(torch.load(classifier_B_save_path))

    extractor.eval()
    classifier_A.eval()
    classifier_B.eval()

    print("Evaluating...")
    target_loader = test_load(t_test, t_gt_test, batch_size)


    A_correct = 0
    B_correct = 0
    A_total = 0
    B_total = 0

    a_prediction = []
    b_prediction = []
    ground_truth = []

    accuracy = []
    total_pool_feature = []
    for index, (t_sample, t_target) in enumerate(target_loader):

        t_sample, t_target = t_sample.to(DEVICE).float(), t_target.to(DEVICE).long()
        t_sample = t_sample.view(-1, AXIS, 1, win_size)
        
        
        output = extractor(t_sample)    
        A_output = classifier_A(output)
        B_output = classifier_B(output)

        _, A_predicted = torch.max(A_output.data, 1)
        A_total += t_target.size(0)
        A_correct += (A_predicted == t_target).sum()

        _, B_predicted = torch.max(B_output.data, 1)
        B_total += t_target.size(0)
        B_correct += (B_predicted == t_target).sum()

        a_prediction.extend(A_predicted.tolist())
        b_prediction.extend(B_predicted.tolist())
        ground_truth.extend(t_target.cpu().numpy().tolist())

    
    acc_test_A = float(A_correct) * 100 / A_total
    acc_test_B = float(B_correct) * 100 / B_total
    
    normalize = True
    plt.figure(figsize=(20, 20))
    a_plot_cm = confusion_matrix(ground_truth, a_prediction)
    plot_confusion_matrix(a_plot_cm, classes=activities, title='Classifier A Confusion matrix'+'Iteration: '+str(e), plot_save_path=plot_save_path)

    a_f1_micro        = f1_score(ground_truth,        a_prediction, average='micro')
    a_precision_micro = precision_score(ground_truth, a_prediction, average='micro')
    a_recall_micro    = recall_score(ground_truth,    a_prediction, average='micro')
    

    plt.figure(figsize=(20, 20))
    b_plot_cm = confusion_matrix(ground_truth, b_prediction)
    plot_confusion_matrix(b_plot_cm, classes=activities, title='Classifier B Confusion matrix'+'Iteration: '+str(e), plot_save_path=plot_save_path)


    b_f1_micro        = f1_score(ground_truth,        b_prediction, average='micro')
    b_precision_micro = precision_score(ground_truth, b_prediction, average='micro')
    b_recall_micro    = recall_score(ground_truth,    b_prediction, average='micro')
    
    file1.write(person_list[source1_user-1] +" "+position_array[position] +" "+ person_list[source2_user-1] +" "+ position_array[position] +" "+ person_list[target_user-1] +" "+ position_array[position] +" "+ str("%.4f"%acc_test_A) +" "+ str("%.4f"%a_f1_micro) +" "+ str("%.4f"%a_precision_micro) +" "+ str("%.4f"%a_recall_micro) +" "+ str("%.4f"%acc_test_B) +" "+ str("%.4f"%b_f1_micro) +" "+ str("%.4f"%b_precision_micro) +" "+ str("%.4f"%b_recall_micro)+"\n")
    file1.close()
    
    plot(plot_save_path, 'A', win_size, overlap, target_user, acc_test_A, plot_common_title, plot_save_path)
    plot(plot_save_path, 'B', win_size, overlap, target_user, acc_test_A, plot_common_title, plot_save_path)


# # Model Execution

# In[26]:
# from DataPreprocess import *

# def DA_Jitter(X, sigma=0.01):
#     myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
#     return X+myNoise

# def DA_Scaling(X, sigma=0.1):
#     scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
#     myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
#     return X*myNoise

# def DA_Rotation(X):
#     axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
#     angle = np.random.uniform(low=-np.pi, high=np.pi)
#     return np.matmul(X, axangle2mat(axis,angle))

# sigma = 0.005
# knot = 2

# ## This example using cubic splice is not the best approach to generate random curves. 
# ## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
# def GenerateRandomCurves(X, sigma=0.2, knot=4):
#     xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
#     yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
#     x_range = np.arange(X.shape[0])
#     cs_x = CubicSpline(xx[:,0], yy[:,0])
#     cs_y = CubicSpline(xx[:,1], yy[:,1])
#     cs_z = CubicSpline(xx[:,2], yy[:,2])
#     return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

# def DistortTimesteps(X, sigma=0.2):
#     tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
#     tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph

#     t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
#     tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
#     tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
#     tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
#     return tt_cum

# def DA_TimeWarp(X, sigma=0.2):
#     tt_new = DistortTimesteps(X, sigma)
#     X_new = np.zeros(X.shape)
#     x_range = np.arange(X.shape[0])

#     X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
#     X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
#     X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
#     return X_new


# class modified_data_loader(Dataset):
#     def __init__(self, samples, labels, transform=None):
#         self.samples = samples
#         self.labels = labels
#         self.transform = transform

#     def __getitem__(self, index):
#         x = np.squeeze(self.samples[index], axis = 1)  # (3,1,128) -> (3,128)
#         x = np.swapaxes(x, 0, 1) # (128,3)
        
#         modified_signal = DA_Jitter(DA_Scaling(DA_TimeWarp(DA_Rotation(x), sigma = 0.2))) 

#         modified_signal = modified_signal.reshape(128,1,3)
#         modified_signal = np.swapaxes(modified_signal, 0, 2) #(1,3,1,128)
        
#         return modified_signal, self.samples[index], self.labels[index]

#     def __len__(self):
#         return len(self.samples)
    
from DataPreprocess import *
from scipy.stats import vonmises

sigma = 0.005
knot = 2
kappa = 0.5

def DA_Jitter(X, sigma=0.01):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
#     angle = np.random.uniform(low=-np.pi, high=np.pi)
    mu, kappa = 0.0, 0.5 # mean and dispersion
    angle = np.random.vonmises(mu, kappa, 1)
    return np.matmul(X, axangle2mat(axis,angle))

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
    tt = GenerateRandomCurves(X, sigma, knot) # Regard these samples aroun 1 as time intervals
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
        
#         modified_signal = DA_Jitter(DA_Scaling(DA_TimeWarp(DA_Rotation(x), sigma = 0.2)), sigma=0.05)  
        modified_signal = DA_Jitter(DA_TimeWarp(DA_Rotation(x), sigma = 0.2), sigma=0.05)  
        modified_signal = modified_signal.reshape(128,1,3)
        modified_signal = np.swapaxes(modified_signal, 0, 2) #(1,3,1,128)
#         print("Get item called")
        
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


if __name__ == '__main__':
    
    S1_train, S1_valid, S1_test = modified_load_train_valid_test(s1_train, s1_gt_train, s1_valid, s1_gt_valid, s1_test, s1_gt_test, batch_size)
    S2_train, S2_valid, S2_test = modified_load_train_valid_test(s2_train, s2_gt_train, s2_valid, s2_gt_valid, s2_test, s2_gt_test, batch_size)
    
    extractor = AccExtractor().to(DEVICE)
    classifier_A = AccClassifier(output_gt_number).to(DEVICE)
    classifier_B = AccClassifier(output_gt_number).to(DEVICE)
    
    param_list = list(extractor.parameters()) + list(classifier_A.parameters()) + list(classifier_B.parameters())
    optimizer = optim.Adam(params=param_list, lr=LEARNING_RATE, betas=(beta1, beta2))
    train(extractor, classifier_A, classifier_B, optimizer, S1_train, S1_valid, S1_test, S2_train, S2_valid, S2_test)

