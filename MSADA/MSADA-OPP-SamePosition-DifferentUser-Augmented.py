#!/usr/bin/env python
# coding: utf-8

# In[1]:


home_url = "/notebooks"


# In[2]:


import sys
sys.path.append(home_url+'/Downloads/ISWC/Debug-Packages/Utils/')
sys.path.append(home_url+'/Downloads/ISWC/Debug-Packages/Model/')

from DataPreprocess import *
from Util import *
from Visualization import *
from loss import *
from msda_classifier import *
from msda_extractor import *
from msda_discriminator import *

# import matplotlib as plt
import pandas as pd
import os
import math
import csv
import pickle
import time
import logging
import seaborn as sns
from pylab import rcParams
import tqdm
import argparse
import random

import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from mpl_toolkits.mplot3d import Axes3D

from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
torch.manual_seed(10)
np.random.seed(10)


# ## Initialization

# In[3]:

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


msada_result = []
s1_losses = []
s2_losses = []
discriminator_losses = []
msada_result = []


# In[4]:

step_size=int(win_size*overlap)
gpu_id = 0
activities = ['Sitting','Standing','Walking','Running']
data_source = {"source1":source1_user, "source2": source2_user, "target":target_user}

DEVICE = torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')


# In[5]:


# url = home_url+ "/Downloads/ISWC/Dataset/PAMAP2_Dataset/Protocol/"
item = ["train","valid","test"]
person_list = ["U1","U2","U3","U4"]
position_array = ["BACK", "RUA", "RLA", "LUA","LLA"]
label = ['Confusion Loss', 'Discriminator Loss', 'Classification Loss']

AXIS = 3
FROM = 0
TO = FROM+3
START = 3
END = 4

steps = 150
batch_size = 32
plot_interval = 100
cm_interval = 20
lr=0.0001
EPSILON = 1e-12
beta1=0.9
beta2=0.999
avg_accuracy_epoch = int(steps*0.15)
plot_common_title = " window: "+ str(win_size) + "--Overlap: "+ str(overlap) + "--Position: "+ position_array[position] + "\n"+ "--Activity: "+ str(activity_num)


# In[6]:


folder_name = str(activity_num)+ " Activity"+"_Window "+str(win_size)+ "_Overlap "+str(overlap)
dataset_path = home_url+"/Downloads/ISWC/Preprocessing/OPPORTUNITY/Data Files/"+folder_name+"/"
user_combination = "/S1: "+str(source1_user)+" S2: "+str(source2_user)+" Target "+ str(target_user)+"/"

common_path = "OPP-SamePosition-DifferentUser-Augmented/"+folder_name+"/Position-"+position_array[position]+user_combination


pretrain_path = home_url+"/Downloads/ISWC/Debug-Pretraining/Pre-trained/"+common_path
plot_save_path = os.getcwd() +"/MSADA/"+common_path
    
if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)


# In[7]:


pretrain_path


# In[8]:


plot_save_path


# In[9]:


experiment_summery_file = os.getcwd() + "/MSADA/OPP-SamePosition-DifferentUser-Augmented/"
file1 = open(experiment_summery_file+"OPP-SamePosition-DifferentUser-Augmented.txt", "a")


# - Data Structure

# In[10]:


s1_train = []
s2_train = []
t_train = []

s1_gt_train = []
s2_gt_train = []
t_gt_train = []


# In[11]:


s1_valid = []
s2_valid = []
t_valid = []

s1_gt_valid = []
s2_gt_valid = []
t_gt_valid = []


# In[12]:


s1_test = []
s2_test = []
t_test = []

s1_gt_test = []
s2_gt_test = []
t_gt_test = []


# ## Data Preprocessing

# In[13]:


for key in data_source:
    for split_index in range(0,3):

        if key == "source1":
            file_name = person_list[source1_user-1] + "_" + position_array[position]+'_'+item[split_index]
            print(file_name)
            df = pd.read_csv(dataset_path+file_name+'.csv', sep=",")  

            if split_index == 0:
                calculate_window(df, s1_train, s1_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, s1_valid, s1_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, s1_test, s1_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
        elif key == "source2":
            file_name = person_list[source2_user-1] + "_" + position_array[position]+'_'+item[split_index]
            print(file_name)
            df = pd.read_csv(dataset_path+file_name+'.csv', sep=",") 

            if split_index == 0:
                calculate_window(df, s2_train, s2_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, s2_valid, s2_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, s2_test, s2_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)
        elif key == "target":
            file_name = person_list[target_user-1] + "_" + position_array[position]+'_'+item[split_index]
            print(file_name)
            df = pd.read_csv(dataset_path+file_name+'.csv', sep=",") 

            if split_index == 0:
                calculate_window(df, t_train, t_gt_train, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 1:
                calculate_window(df, t_valid, t_gt_valid, win_size, step_size, FROM, TO, START, END, AXIS)
            elif split_index == 2:
                calculate_window(df, t_test, t_gt_test, win_size, step_size, FROM, TO, START, END, AXIS)


# In[14]:


s1_train = np.concatenate( s1_train, axis=0 ).astype(np.float32)
s1_gt_train = np.array(s1_gt_train  ).astype(np.float32)
s1_valid = np.concatenate( s1_valid, axis=0 ).astype(np.float32)
s1_gt_valid = np.array(s1_gt_valid  ).astype(np.float32)
s1_test = np.concatenate( s1_test, axis=0 ).astype(np.float32)
s1_gt_test = np.array(s1_gt_test  ).astype(np.float32)


# In[15]:


s2_train = np.concatenate( s2_train, axis=0 ).astype(np.float32)
s2_gt_train = np.array(s2_gt_train  ).astype(np.float32)
s2_valid = np.concatenate( s2_valid, axis=0 ).astype(np.float32)
s2_gt_valid = np.array(s2_gt_valid  ).astype(np.float32)
s2_test = np.concatenate( s2_test, axis=0 ).astype(np.float32)
s2_gt_test = np.array(s2_gt_test  ).astype(np.float32)


# In[16]:


t_train = np.concatenate( t_train, axis=0 ).astype(np.float32)
t_gt_train = np.array(t_gt_train  ).astype(np.float32)
t_valid = np.concatenate( t_valid, axis=0 ).astype(np.float32)
t_gt_valid = np.array(t_gt_valid  ).astype(np.float32)
t_test = np.concatenate( t_test, axis=0 ).astype(np.float32)
t_gt_test = np.array(t_gt_test  ).astype(np.float32)


# In[17]:


plot_training_data_distribution_multi_source(s1_gt_train, s2_gt_train, t_gt_train, activities, plot_save_path)


# In[18]:


output_gt_number = len(np.unique(s1_gt_train))


# In[19]:


source1_train_loader, source1_valid_loader, source1_test_loader = modified_load_train_valid_test(s1_train, s1_gt_train, s1_valid, s1_gt_valid, s1_test, s1_gt_test, batch_size)
source2_train_loader, source2_valid_loader, source2_test_loader = modified_load_train_valid_test(s2_train, s2_gt_train, s2_valid, s2_gt_valid, s2_test, s2_gt_test, batch_size)
target_train_loader, target_valid_loader, target_test_loader = modified_load_train_valid_test(t_train, t_gt_train, t_valid, t_gt_valid, t_test, t_gt_test, batch_size)


# ## Load the pretrained networks

# In[20]:


extractor = AccExtractor().to(DEVICE)
classifier_A = AccClassifier(output_gt_number).to(DEVICE)
classifier_B = AccClassifier(output_gt_number).to(DEVICE)

discriminator_A = AccDiscriminator().to(DEVICE)
discriminator_B = AccDiscriminator().to(DEVICE)

extractor.load_state_dict(torch.load(pretrain_path + "extractor.pth"))
classifier_A.load_state_dict(torch.load(pretrain_path + "classifierA.pth"))
classifier_B.load_state_dict(torch.load(pretrain_path + "classifierB.pth"))


# In[21]:


optim_extract = optim.Adam(extractor.parameters(), lr=lr, betas=(beta1, beta2))

# weight decay is making discriminators stronger.
optim_s1_t_dis = optim.Adam(discriminator_A.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-2)
optim_s2_t_dis = optim.Adam(discriminator_B.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-2)
optim_s1_cls = optim.Adam(classifier_A.parameters(), lr=lr, betas=(beta1, beta2))
optim_s2_cls = optim.Adam(classifier_B.parameters(), lr=lr, betas=(beta1, beta2))


# ## MSADA

# In[22]:


start_time = time.time()
loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
calculate_softmax = nn.Softmax(dim=0)


source1_accuracy_sum = 0
source2_accuracy_sum = 0
target_accuracy_sum = 0


source1_accuracy_list = [0] * avg_accuracy_epoch
source2_accuracy_list = [0] * avg_accuracy_epoch
target_accuracy_list = [0] * avg_accuracy_epoch
source1_running_accuracy = 0
source2_running_accuracy = 0
target_running_accuracy = 0


for step in tqdm.tqdm(range(steps)):

    print("Step: {} #### Part1: Multi-way Adversarial Adaptation".format(step))
    
    # Features fed into the classifiers and domain discriminators. # C fixed, F and D are updated
    extractor.train()
    discriminator_A.train()
    discriminator_B.train()
    classifier_A.eval()
    classifier_B.eval()  

    # For two sources, two weighted losses
    s1_weight_loss = 0
    s2_weight_loss = 0
    
    s1_classification_loss, s2_classification_loss = 0, 0
    s1_discriminator_loss, s2_discriminator_loss = 0, 0
    s1_confusion_loss, s2_confusion_loss = 0, 0
    
    s1_prediction = []
    s2_prediction = []
    s1_t_prediction = []
    s2_t_prediction = []
    
    min_dataloader_len = min(len(source1_train_loader), len(source2_train_loader), len(target_train_loader))
    s1_loader, s2_loader, t_loader = iter(source1_train_loader), iter(source2_train_loader), iter(target_train_loader)

    # Target has smaller data points. It will finish iterating first.
    for i, (Modified_t_sample, t_sample, t_labels) in tqdm.tqdm(enumerate(t_loader)):
        try:
            Modified_s1_Sample, s1_sample, s1_labels = s1_loader.next()
        except StopIteration:
            s1_loader = iter(source1_train_loader)
            Modified_s1_Sample, s1_sample, s1_labels = s1_loader.next()

        try:
            Modified_s2_Sample, s2_sample, s2_labels = s2_loader.next()
        except StopIteration:
            s2_loader = iter(source2_train_loader)
            Modified_s2_Sample, s2_sample, s2_labels = s2_loader.next()

        s1_sample, s1_labels = Variable(s1_sample.cuda(gpu_id)), Variable(s1_labels.cuda(gpu_id))
        s2_sample, s2_labels = Variable(s2_sample.cuda(gpu_id)), Variable(s2_labels.cuda(gpu_id))
        t_sample = Variable(t_sample.cuda(gpu_id))
        
        Modified_s1_Sample = Variable(Modified_s1_Sample.float().cuda(gpu_id))
        Modified_s2_Sample = Variable(Modified_s2_Sample.float().cuda(gpu_id))
        Modified_t_sample = Variable(Modified_t_sample.float().cuda(gpu_id))

        s1_feature = extractor(s1_sample)
        s2_feature = extractor(s2_sample)
        t_feature = extractor(t_sample)
        
        Modified_s1_feature = extractor(Modified_s1_Sample)
        Modified_s2_feature = extractor(Modified_s2_Sample)
        Modified_t_feature = extractor(Modified_t_sample)
        

        # Classification loss. Second part of the Equation-4
        s1_cls = classifier_A(s1_feature)
        s2_cls = classifier_B(s2_feature)
        Modified_s1_cls = classifier_A(Modified_s1_feature)
        Modified_s2_cls = classifier_B(Modified_s2_feature)
        
        s1_labels = s1_labels.long()
        s2_labels = s2_labels.long()

        s1_cls_loss = get_cls_loss(s1_cls, s1_labels)
        s2_cls_loss = get_cls_loss(s2_cls, s2_labels)
        Modified_s1_cls_loss = get_cls_loss(Modified_s1_cls, s1_labels)
        Modified_s2_cls_loss = get_cls_loss(Modified_s2_cls, s2_labels)

        s1_source = discriminator_A(s1_feature)
        Modified_s1_source = discriminator_A(Modified_s1_feature)
        s1_target = discriminator_A(t_feature)
        Modified_s1_target = discriminator_A(Modified_t_feature)
        s2_source = discriminator_B(s2_feature)
        Modified_s2_source = discriminator_B(Modified_s2_feature)
        s2_target = discriminator_B(t_feature)
        Modified_s2_target = discriminator_B(Modified_t_feature)

        # Calculate confusion loss
        s1_s_conf_loss = get_confusion_loss(s1_source)
        s1_tar_conf_loss = get_confusion_loss(s1_target)
        s1_t_confusion_loss = -(0.5 * s1_s_conf_loss + 0.5 * s1_tar_conf_loss)
        
        Modified_s1_s_conf_loss = get_confusion_loss(Modified_s1_source)
        Modified_s1_tar_conf_loss = get_confusion_loss(Modified_s1_target)
        Modified_s1_t_confusion_loss = -(0.5 * Modified_s1_s_conf_loss + 0.5 * Modified_s1_tar_conf_loss)
        
        s2_s_conf_loss = get_confusion_loss(s2_source)
        s2_tar_conf_loss = get_confusion_loss(s2_target)
        s2_t_confusion_loss = -(0.5 * s2_s_conf_loss + 0.5 * s2_tar_conf_loss)
        
        Modified_s2_s_conf_loss = get_confusion_loss(Modified_s2_source)
        Modified_s2_tar_conf_loss = get_confusion_loss(Modified_s2_target)
        Modified_s2_t_confusion_loss = -(0.5 * Modified_s2_s_conf_loss + 0.5 * Modified_s2_tar_conf_loss)

        # Perplexity Score
        s1_weight_loss += -torch.mean(torch.log(1-s1_target)).data
        s1_weight_loss += -torch.mean(torch.log(1-Modified_s1_target)).data
        s2_weight_loss += -torch.mean(torch.log(1-s2_target)).data
        s2_weight_loss += -torch.mean(torch.log(1-Modified_s2_target)).data


        # NOTE: If all the parameters are going to be updated then optimizer.zero_grad() == Network.zero_grad()
        discriminator_A.zero_grad()
        discriminator_B.zero_grad()


        # Detach features so that it does not update the feature extractor. Only update the discriminators
        # Recalculated with the detach method so that the backpropagation does not accumulate gradient to feature extractor
        s1_source = discriminator_A(s1_feature.detach())
        s1_target = discriminator_A(t_feature.detach())
        s2_source = discriminator_B(s2_feature.detach())
        s2_target = discriminator_B(t_feature.detach())
        
        Modified_s1_source = discriminator_A(Modified_s1_feature.detach())
        Modified_s1_target = discriminator_A(Modified_t_feature.detach())
        Modified_s2_source = discriminator_B(Modified_s2_feature.detach())
        Modified_s2_target = discriminator_B(Modified_t_feature.detach())
        

        # Discrepency loss(Target 0, Source 1)-Determines which source discriminator confusion loss to use for updating feature extractor
        s1_t_dis_loss = get_dis_loss(s1_source, s1_target)
        s2_t_dis_loss = get_dis_loss(s2_source, s2_target)
        
        Modified_s1_t_dis_loss = get_dis_loss(Modified_s1_source, Modified_s1_target)
        Modified_s2_t_dis_loss = get_dis_loss(Modified_s2_source, Modified_s2_target)

        # Update D
        # Cross check the update process. Chech its value before and after this statement.
        torch.autograd.backward([s1_t_dis_loss, s2_t_dis_loss, Modified_s1_t_dis_loss, Modified_s2_t_dis_loss])
        optim_s1_t_dis.step()
        optim_s2_t_dis.step()
        

        # Identify the most discriminant source. Then take the confusion loss from that domain.
        # Update the generator based on classification and confusion loss
        extractor.zero_grad()
        if s1_t_dis_loss.data >= s2_t_dis_loss.data:
            SELECTIVE_SOURCE = "S1"
            # Why the confusion effect of the other discriminator is not considered?
            torch.autograd.backward([s1_cls_loss, s2_cls_loss, s1_t_confusion_loss, Modified_s1_t_confusion_loss])
        else:
            SELECTIVE_SOURCE = "S2"
            torch.autograd.backward([s1_cls_loss, s2_cls_loss, s2_t_confusion_loss, Modified_s2_t_confusion_loss])

        optim_extract.step()
        
        # Batchwise sum
        s1_prediction.append(np.sum(s1_source.data.cpu().numpy().tolist())/len(s1_source.data.cpu().numpy().tolist()))
        s2_prediction.append(np.sum(s2_source.data.cpu().numpy().tolist())/len(s2_source.data.cpu().numpy().tolist()))
        s1_t_prediction.append(np.sum(s1_target.data.cpu().numpy().tolist())/len(s1_target.data.cpu().numpy().tolist()))
        s2_t_prediction.append(np.sum(s2_target.data.cpu().numpy().tolist())/len(s2_target.data.cpu().numpy().tolist()))

        
        s1_classification_loss += s1_cls_loss
        s2_classification_loss += s2_cls_loss
        s1_discriminator_loss += s1_t_dis_loss
        s2_discriminator_loss += s2_t_dis_loss
        s1_confusion_loss += s1_t_confusion_loss
        s2_confusion_loss += s2_t_confusion_loss

    # Both Classifier loss.
    s1_weight = s1_weight_loss / (s1_weight_loss + s2_weight_loss)
    s2_weight = s2_weight_loss / (s1_weight_loss + s2_weight_loss)

    s1_weight = s1_weight.cpu().data.numpy()
    s2_weight = s2_weight.cpu().data.numpy()
          
#     s1_losses.append([s1_classification_loss, s1_discriminator_loss, s1_confusion_loss])
#     s2_losses.append([s2_classification_loss, s2_discriminator_loss, s2_confusion_loss])
    
#     np.savetxt(plot_save_path+'source1_result.csv', np.array(s1_losses), fmt='%.4f', delimiter=',')
#     np.savetxt(plot_save_path+'source2_result.csv', np.array(s2_losses), fmt='%.4f', delimiter=',')
    
#     s1_result = sum(s1_prediction)/len(s1_prediction)
#     s1_t_result = sum(s1_t_prediction)/len(s1_t_prediction)
#     s2_result = sum(s2_prediction)/len(s2_prediction)
#     s2_t_result = sum(s2_t_prediction)/len(s2_t_prediction)
#     discriminator_losses.append([s1_result, s1_t_result, s2_result, s2_t_result, s1_weight, s2_weight])
#     np.savetxt(plot_save_path+'discriminator_result.csv', np.array(discriminator_losses), fmt='%.4f', delimiter=',')
        
        
#     # Validation
#     print("Step: {} #### Part2: Validation".format(step))
    
#     extractor.eval()
#     classifier_A.eval()
#     classifier_B.eval()

#     target_correct = 0
#     source1_correct = 0
#     source2_correct = 0
    
#     min_valid_loader_len = min(len(source1_valid_loader), len(source2_valid_loader), len(target_valid_loader))
#     s1_valid_loader, s2_valid_loader, t_valid_loader = iter(source1_valid_loader), iter(source2_valid_loader), iter(target_valid_loader)
    
#     # Target Vallidation
#     for (modified_sample, sample, labels) in t_valid_loader:
#         sample = Variable(sample.cuda(gpu_id))  
#         sample_feature = extractor(sample)

#         s1_cls = classifier_A(sample_feature)
#         s2_cls = classifier_B(sample_feature)
#         s1_cls = s1_cls.data.cpu().numpy()
#         s2_cls = s2_cls.data.cpu().numpy()

#         # s1_weight and s2_weight should be generated from the discriminator
#         res = s1_cls * s1_weight + s2_cls * s2_weight

#         pred = res.argmax(axis=1)
#         labels = labels.numpy()
#         target_correct += np.equal(labels, pred).sum()
        
#     total_item = int(len(t_valid_loader.dataset)/batch_size)*batch_size
#     target_accuracy = target_correct/total_item
    

#     # Source1 Vallidation
#     for (modified_sample, sample, labels) in s1_valid_loader:
#         sample = Variable(sample.cuda(gpu_id))
#         sample_feature = extractor(sample)

#         s1_cls = classifier_A(sample_feature)
#         s1_cls = s1_cls.data.cpu().numpy()

#         pred = s1_cls.argmax(axis=1)
#         labels = labels.numpy() 
#         source1_correct += np.equal(labels, pred).sum()
        
#     total_item = int(len(s1_valid_loader.dataset)/batch_size)*batch_size
#     source1_accuracy = source1_correct/total_item
    
    
#     # Source2 Vallidation
#     for (modified_sample, sample, labels) in s2_valid_loader:
#         sample = Variable(sample.cuda(gpu_id))
#         sample_feature = extractor(sample)

#         s2_cls = classifier_B(sample_feature)
#         s2_cls = s2_cls.data.cpu().numpy()

#         pred = s2_cls.argmax(axis=1)
#         labels = labels.numpy()
#         source2_correct += np.equal(labels, pred).sum()
    
#     total_item = int(len(s2_valid_loader.dataset)/batch_size)*batch_size
#     source2_accuracy = source2_correct/total_item   
        
#     msada_result.append([target_accuracy, source1_accuracy, source2_accuracy])
#     np.savetxt(plot_save_path+'msada_result.csv', np.array(msada_result), fmt='%.4f', delimiter=',')
    
#     source1_accuracy_list[step%avg_accuracy_epoch] = source1_accuracy
#     source2_accuracy_list[step%avg_accuracy_epoch] = source2_accuracy
#     target_accuracy_list[step%avg_accuracy_epoch] = target_accuracy

#     source1_current_accuracy = sum(source1_accuracy_list)/avg_accuracy_epoch
#     source2_current_accuracy = sum(source2_accuracy_list)/avg_accuracy_epoch
#     target_current_accuracy = sum(target_accuracy_list)/avg_accuracy_epoch
    
#     if source1_current_accuracy > source1_running_accuracy:
#         source1_running_accuracy = source1_current_accuracy
        
#     if source2_current_accuracy > source2_running_accuracy:
#         source2_running_accuracy = source2_current_accuracy
        
#     if target_current_accuracy > target_running_accuracy:
#         target_running_accuracy = target_current_accuracy
        
        
    # Final # of accuracy
#     if step >= steps-avg_accuracy_epoch-1:
#         source1_accuracy_sum += source1_accuracy
#         source2_accuracy_sum += source2_accuracy
#         target_accuracy_sum += target_accuracy

#         final_s1_accuracy = source1_accuracy_sum/avg_accuracy_epoch
#         final_s2_accuracy = source2_accuracy_sum/avg_accuracy_epoch
#         final_tar_accuracy = target_accuracy_sum/avg_accuracy_epoch
    
    if step == steps-1:
        
        extractor.eval()
        classifier_A.eval()
        classifier_B.eval()

        prediction = []
        ground_truth = []
        
        target_total, target_correct = 0, 0
        
        # Target domain Test Split
        target_test_itr = iter(target_test_loader)
        for (modified_sample, sample, labels) in target_test_itr:
            sample = Variable(sample.cuda(gpu_id))  
            sample_feature = extractor(sample)

            s1_cls = classifier_A(sample_feature)
            s2_cls = classifier_B(sample_feature)
            s1_cls = s1_cls.data.cpu().numpy()
            s2_cls = s2_cls.data.cpu().numpy()

            # s1_weight and s2_weight should be generated from the discriminator
            res = s1_cls * s1_weight + s2_cls * s2_weight

            pred = res.argmax(axis=1)
            labels = labels.numpy()
            
            print(labels.shape)
            target_total += int(labels.shape[0])
            target_correct += (pred == labels).sum()
            
            prediction.extend(pred.tolist())
            ground_truth.extend(labels.tolist())
            
        target_accuracy = float(target_correct) / target_total   
        print("Total item: "+ str(target_total) + " Total Correct: "+ str(target_correct))
        
        plt.figure(figsize=(13,10))
        plot_cm = confusion_matrix(ground_truth, prediction)
        plot_confusion_matrix(plot_cm, classes=activities,
                              title='Confusion matrix-Iteration: '+str(step)+ 'Accuracy: '+str(target_accuracy), plot_save_path=plot_save_path)
        
        f1_micro        = f1_score(ground_truth,        prediction, average='micro')
        precision_micro = precision_score(ground_truth, prediction, average='micro')
        recall_micro    = recall_score(ground_truth,    prediction, average='micro')
        
        file1.write(person_list[source1_user-1] +" "+position_array[position] +" "+ person_list[source2_user-1] +" "+                    position_array[position] +" "+ person_list[target_user-1] +" "+ position_array[position] +" "+                    str("%.4f"%target_accuracy) +" "+ str("%.4f"%f1_micro) +" "+ str("%.4f"%precision_micro) +" "+                    str("%.4f"%recall_micro) +"\n")
        file1.close()
    
         
# new_accuracy_plot(target_user, plot_save_path, source1_running_accuracy, source2_running_accuracy, target_running_accuracy)
# new_loss_plot(target_user, plot_save_path)


# In[ ]:




