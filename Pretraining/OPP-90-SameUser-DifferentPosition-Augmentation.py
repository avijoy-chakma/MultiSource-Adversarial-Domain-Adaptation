#!/usr/bin/env python
# coding: utf-8

# In[1]:


directory = "POC"
main_folder = "OPP-90-SameUser-DifferentPosition-Augmentation"
base_url = "/notebooks/Downloads/"+directory


# In[2]:


import sys
sys.path.append(base_url+'/Debug-Packages/Utils/')
sys.path.append(base_url+'/Debug-Packages/Model/')

from msda_classifier import *
from msda_extractor import *
from msda_discriminator import *
from msda_convolution import *
from DataPreprocess import *

# from msda_classifier import *
# from msda_extractor import *
# from msda_discriminator import *
# from DataPreprocess import *
from Util import *
from Visualization import *

import pandas as pd
import os
import numpy as np
import math
import csv
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn as sns
from pylab import rcParams
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
from msda_extractor import AccExtractor
from msda_classifier import AccClassifier
from matplotlib.lines import Line2D   
from tqdm import tqdm
import tqdm
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
torch.manual_seed(10)
np.random.seed(10)


# # Initialization

# In[3]:

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--win_size")
parser.add_argument("--overlap")
parser.add_argument("--source1")
parser.add_argument("--source2")
parser.add_argument("--target")
parser.add_argument("--activity")
parser.add_argument("--user")


args = parser.parse_args()

win_size=int(args.win_size)
overlap = float(args.overlap)
source1 = int(args.source1)
source2 = int(args.source2)
target = int(args.target)
activity_num = int(args.activity)
selected_user = int(args.user)


step_size=int(win_size*overlap)
gpu_id= 0

activities = ['Sitting','Standing','Walking','Running']
data_source = {"source1":source1, "source2": source2, "target":target}


# In[4]:


item = ["train","valid","test"]
person_list = ["U1","U2","U3","U4"]
position_array = ["BACK", "RUA", "RLA", "LUA","LLA"]

train_percentage = 0.6
valid_percentage = 0.2


AXIS = 3
FROM = 0
TO = FROM+3
START = 3
END = 4

batch_size = 32
N_EPOCH = 100
LEARNING_RATE = 0.00001
beta1=0.9
beta2=0.99

plot_common_title = " window: "+ str(win_size) + "--Overlap: "+ str(overlap) + "--Source1:"+position_array[data_source["source1"]] + "\n"+"--Source2: "+ position_array[data_source["source2"]] + "--Target: "+ position_array[data_source["target"]] + "\n"+ "--Activity: "+ str(activity_num)


# In[5]:


dataset_settings = str(activity_num)+ " Activity"+"_Window "+str(win_size)+ "_Overlap "+str(overlap)
dataset_path = base_url +'/Preprocessing/OPPORTUNITY/Data Files/'+dataset_settings+'/'


save_path = os.getcwd() + "/Pre-trained/"+main_folder+"/"+dataset_settings+"/Target-Position-"+position_array[data_source["target"]]+"/User "+ str(selected_user)+"/"


if not os.path.exists(save_path):
    os.makedirs(save_path)

# In[6]:

file1 = open(os.getcwd() + "/Pre-trained/"+main_folder+"/"+main_folder+".txt", "a")


feature_save_path = save_path + "extractor.pth"
classifier_A_save_path = save_path + "classifierA.pth"
classifier_B_save_path = save_path + "classifierB.pth"


# In[9]:


DEVICE = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')
A_result = []
B_result = []


# - Data Structure

# In[10]:


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

# In[11]:


for key in data_source:
    for split_index in range(0,3):
        file_name = person_list[selected_user-1] + "_" + position_array[data_source[key]]+'_'+item[split_index]

        df = pd.read_csv(dataset_path+file_name+'.csv', sep=",")   

        if key == "source1":
            if split_index == 0:
                calculate_window(df, s1_train, s1_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, s1_valid, s1_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, s1_test, s1_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
        elif key == "source2":
            if split_index == 0:
                calculate_window(df, s2_train, s2_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, s2_valid, s2_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, s2_test, s2_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
        elif key == "target":
            if split_index == 0:
                calculate_window(df, t_train, t_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, t_valid, t_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, t_test, t_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)


# In[12]:


s1_train = np.concatenate( s1_train, axis=0 ).astype(np.float32)
s1_gt_train = np.array(s1_gt_train  ).astype(np.float32)
s1_valid = np.concatenate( s1_valid, axis=0 ).astype(np.float32)
s1_gt_valid = np.array(s1_gt_valid  ).astype(np.float32)
s1_test = np.concatenate( s1_test, axis=0 ).astype(np.float32)
s1_gt_test = np.array(s1_gt_test  ).astype(np.float32)


# In[13]:


s2_train = np.concatenate( s2_train, axis=0 ).astype(np.float32)
s2_gt_train = np.array(s2_gt_train  ).astype(np.float32)
s2_valid = np.concatenate( s2_valid, axis=0 ).astype(np.float32)
s2_gt_valid = np.array(s2_gt_valid  ).astype(np.float32)
s2_test = np.concatenate( s2_test, axis=0 ).astype(np.float32)
s2_gt_test = np.array(s2_gt_test  ).astype(np.float32)


# In[14]:


t_train = np.concatenate( t_train, axis=0 ).astype(np.float32)
t_gt_train = np.array(t_gt_train  ).astype(np.float32)
t_valid = np.concatenate( t_valid, axis=0 ).astype(np.float32)
t_gt_valid = np.array(t_gt_valid  ).astype(np.float32)
t_test = np.concatenate( t_test, axis=0 ).astype(np.float32)
t_gt_test = np.array(t_gt_test  ).astype(np.float32)


# In[15]:


plot_training_data_distribution_multi_source(s1_gt_train, s2_gt_train, t_gt_train, activities, save_path)


# In[16]:


output_gt_number = len(np.unique(s1_gt_train))


# # Model Training

# In[17]:


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
        
        for index, (Aug_A_Sample, A_sample, A_target) in enumerate(S1_train):
            
            try:
                Aug_B_Sample, B_sample, B_target = B_train_itr.next()
            except StopIteration:
                B_train_itr = iter(S2_train)
                Aug_B_Sample, B_sample, B_target = B_train_itr.next()

            
            Aug_A_Sample, A_sample, A_target = Aug_A_Sample.to(DEVICE).float(), A_sample.to(DEVICE).float(), A_target.to(DEVICE).long()        
            A_sample = A_sample.view(-1, AXIS, 1, win_size)
            Aug_A_Sample = Aug_A_Sample.view(-1, AXIS, 1, win_size)
            
            
            
            Aug_B_Sample, B_sample, B_target = Aug_B_Sample.to(DEVICE).float(), B_sample.to(DEVICE).float(), B_target.to(DEVICE).long()        
            B_sample = B_sample.view(-1, AXIS, 1, win_size)
            Aug_B_Sample = Aug_B_Sample.view(-1, AXIS, 1, win_size)
            

            
            A_feature, indices1, indices2, shape1, shape2 = extractor(A_sample)
            A_output = classifier_A(A_feature)
            
            Aug_A_feature, indices1, indices2, shape1, shape2 = extractor(Aug_A_Sample)
            Aug_A_output = classifier_A(Aug_A_feature)
            
            B_feature, indices1, indices2, shape1, shape2 = extractor(B_sample)
            B_output = classifier_B(B_feature)
            
            Aug_B_feature, indices1, indices2, shape1, shape2 = extractor(Aug_B_Sample)
            Aug_B_output = classifier_B(Aug_B_feature)
            
            
            
            A_loss = criterion(A_output, A_target)
            B_loss = criterion(B_output, B_target)
            
            Aug_A_loss = criterion(Aug_A_output, A_target)
            Aug_B_loss = criterion(Aug_B_output, B_target)

            A_train_loss += A_loss
            B_train_loss += B_loss
            
            
            # Cummulative Loss
            total_loss = sum([A_loss, B_loss, Aug_A_loss, Aug_B_loss])
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
            
            
        A_acc_train = float(A_correct) * 100/ A_total
        B_acc_train = float(B_correct) * 100/ B_total


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
            
            for valid_index, (Aug_A_sample, A_sample, A_target) in enumerate(S1_valid):
                try:
                    Aug_B_sample, B_sample, B_target = B_valid_itr.next()
                except StopIteration:
                    B_valid_itr = iter(S2_valid)
                    Aug_B_sample, B_sample, B_target = B_valid_itr.next()
                    
                
                Aug_A_Sample, A_sample, A_target = Aug_A_Sample.to(DEVICE).float(), A_sample.to(DEVICE).float(), A_target.to(DEVICE).long()        
                A_sample = A_sample.view(-1, AXIS, 1, win_size)
                Aug_A_Sample = Aug_A_Sample.view(-1, AXIS, 1, win_size)

                Aug_B_Sample, B_sample, B_target = Aug_B_Sample.to(DEVICE).float(), B_sample.to(DEVICE).float(), B_target.to(DEVICE).long()        
                B_sample = B_sample.view(-1, AXIS, 1, win_size)
                Aug_B_Sample = Aug_B_Sample.view(-1, AXIS, 1, win_size)

                
                feature, indices1, indices2, shape1, shape2 = extractor(A_sample)    
                A_output = classifier_A(feature)
                
                feature, indices1, indices2, shape1, shape2 = extractor(B_sample)
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
    
        A_acc_valid = float(A_correct) * 100/ A_total
        B_acc_valid = float(B_correct) * 100/ B_total
        
        if e % 20 == 0:
            tqdm.tqdm.write('Epoch: [{}/{}], A Train Acc: {:.2f}%, B Train Acc: {:.2f}%'.format(e + 1, N_EPOCH, A_acc_train, B_acc_train))
            tqdm.tqdm.write('Epoch: [{}/{}], A Valid Acc: {:.2f}%, B Valid Acc: {:.2f}%'.format(e + 1, N_EPOCH, A_acc_valid, B_acc_valid))
        
        
        torch.save(extractor.state_dict(),feature_save_path)
        torch.save(classifier_A.state_dict(),classifier_A_save_path)
        torch.save(classifier_B.state_dict(),classifier_B_save_path)
        
    evaluation(e)


# # Target Domain Test Dataset Evaluation

# In[18]:


def evaluation(e):
    extractor = AccConvolution().to(DEVICE)
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
        
        
        output, indices1, indices2, shape1, shape2 = extractor(t_sample)    
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
    plot_confusion_matrix(a_plot_cm, classes=activities, title='Classifier A Confusion matrix'+'Iteration: '+str(e), plot_save_path=save_path)

    a_f1_micro        = f1_score(ground_truth,        a_prediction, average='micro')
    a_precision_micro = precision_score(ground_truth, a_prediction, average='micro')
    a_recall_micro    = recall_score(ground_truth,    a_prediction, average='micro')
    

    plt.figure(figsize=(20, 20))
    b_plot_cm = confusion_matrix(ground_truth, b_prediction)
    plot_confusion_matrix(b_plot_cm, classes=activities, title='Classifier B Confusion matrix'+'Iteration: '+str(e), plot_save_path=save_path)


    b_f1_micro        = f1_score(ground_truth,        b_prediction, average='micro')
    b_precision_micro = precision_score(ground_truth, b_prediction, average='micro')
    b_recall_micro    = recall_score(ground_truth,    b_prediction, average='micro')
    
    
    file1.write(person_list[selected_user-1] +" "+position_array[source1] +" "+ person_list[selected_user-1] +" "+ position_array[source2] +" "+ person_list[selected_user-1] +" "+ position_array[target] +" "+ str("%.4f"%acc_test_A) +" "+ str("%.4f"%a_f1_micro) +" "+ str("%.4f"%a_precision_micro) +" "+ str("%.4f"%a_recall_micro) +" "+ str("%.4f"%acc_test_B) +" "+ str("%.4f"%b_f1_micro) +" "+ str("%.4f"%b_precision_micro) +" "+ str("%.4f"%b_recall_micro)+"\n")
    file1.close()
   
    
#     plot(save_path, 'A', win_size, overlap, selected_user, acc_test_A, plot_common_title, save_path)
#     plot(save_path, 'B', win_size, overlap, selected_user, acc_test_A, plot_common_title, save_path)


# In[19]:


if __name__ == '__main__':
    
    S1_train, S1_valid, S1_test = modified_load_train_valid_test(s1_train, s1_gt_train, s1_valid, s1_gt_valid, s1_test, s1_gt_test, batch_size)
    S2_train, S2_valid, S2_test = modified_load_train_valid_test(s2_train, s2_gt_train, s2_valid, s2_gt_valid, s2_test, s2_gt_test, batch_size)
    
    extractor = AccConvolution().to(DEVICE)
    classifier_A = AccClassifier(output_gt_number).to(DEVICE)
    classifier_B = AccClassifier(output_gt_number).to(DEVICE)
    
    param_list = list(extractor.parameters()) + list(classifier_A.parameters()) + list(classifier_B.parameters())
    optimizer = optim.Adam(params=param_list, lr=LEARNING_RATE, betas=(beta1, beta2))
    train(extractor, classifier_A, classifier_B, optimizer, S1_train, S1_valid, S1_test, S2_train, S2_valid, S2_test)


# In[ ]:




