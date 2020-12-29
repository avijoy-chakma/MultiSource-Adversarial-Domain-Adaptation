
# coding: utf-8

# In[1]:


# https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition


# In[2]:


import sys
sys.path.append('/home/avijoy/Downloads/DTCN-AR/Packages/Utils/')
import matplotlib as plt
import pandas as pd
import os
import numpy as np
import math
import csv
from DataPreprocess import standardization_and_normalization
# from pandas.compat import StringIO
# from pandas_datareader import data
from sklearn import preprocessing
# from tqdm import tqdm
import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--win_size")
parser.add_argument("--window_overlap")
parser.add_argument("--source1")
parser.add_argument("--source2")
parser.add_argument("--target")
parser.add_argument("--activity")

args = parser.parse_args()



win_size=int(args.win_size)
overlap = float(args.window_overlap)
source1 = int(args.source1)
source2 = int(args.source2)
target = int(args.target)
activity_num = int(args.activity)

# In[3]:


url = "/home/avijoy/Downloads/DTCN-AR/Dataset/OpportunityUCIDataset/dataset/"


# In[4]:


# save_path = "/home/avijoy/Downloads/DTCN-AR/Preprocessing/OPPORTUNITY/ACC-Position-Preprocessed/"


# In[5]:


# win_size=64
# overlap = 0.5
# activity_num = 4
# source1 = 0
# source2 = 1
# target = 2

step_size=int(win_size*overlap)
selected_user = 1
train_percentage = 0.6
valid_percentage = 0.2


AXIS = 3
FROM = 0
TO = FROM+3
START = 3
END = 4


# In[6]:


item = ["train","valid","test"]
user = ["user1","user2","user3","user4"]
position = ["back", "RUA", "RLA", "LUA","LLA"]


# In[7]:


folder_name = str(activity_num)+ " Activity"+"_Window "+str(win_size)+ "_Overlap "+str(overlap)
save_path = os.getcwd() +"/Data Files/"+folder_name+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)


# # OLD and dataset accelerometer values are in thousends 
#     - Locomotion Annotation: Stand, Walk, Sit, Lie
#     Column Information:
#     1: Milisec
#     38-46: Back IMU
#     51-59: Right Upper Arm
#     64-72: Right Lower Arm
#     77-85: Left Upper Arm
#     90-98: Left Lower Arm
#     103-117: Left Shoe
#     119-133: Right Shoe
#     244: Locomotion
#     
#     Shoe accumulated data is different than the Back, Left, right arm. Leave shoe data seperate.

# ### Directory Structure
#     - 4 user
#     - 5 runs for each user

# ### Locomotion Information
#     1   -   Locomotion   -   Stand
#     2   -   Locomotion   -   Walk
#     4   -   Locomotion   -   Sit
#     5   -   Locomotion   -   Lie

# ### Label Mapping
#     - Stand: 1
#     - Walk: 3
#     - Sit: 0
#     - Lie: 2

# In[8]:


total_frame = {}
BACK_frame = {}
RUA_frame = {}
RLA_frame = {}
LUA_frame = {}
LLA_frame = {}


# In[9]:


# Back_frame, RUA_frame, RLA_frame, LUA_frame, LLA_frame: 
# 11-15: User1 : For all 5 Frames
# 21-25: User2
# 31-35: User3
# 41-45: User4


# In[10]:


# def plot_activity(df, position='Default'):
#     data = df[['Acc_x', 'Acc_y','Acc_z']][:df.shape[0]]

# #     position += '  Index Range: '+str(df.index[0])+ '  to  '+ str(df.index[-1])
#     axis = data.plot(subplots=True, figsize=(16, 12), title=position)
#     for ax in axis:
#         ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
# #         ax.figure.savefig(visualization_url+position+'.png')


# In[11]:


for user_index in range(1,5):
    for run_index in range(1,6):
        file_name = "S"+str(user_index)+"-ADL"+str(run_index)+".dat"
        dataframe = pd.read_csv(url+file_name,sep=" ", header=None) 
        
        dataframe.sort_values(0)
        index = (user_index * 10) + run_index
        
        BACK_IMU_frame = dataframe[[37,38,39,243]].copy()
        BACK_IMU_frame.rename(columns={37: 'Acc_x', 38:'Acc_y', 39: 'Acc_z', 243: 'gt'}, inplace=True)
        BACK_IMU_frame = BACK_IMU_frame.dropna()
        BACK_IMU_frame = BACK_IMU_frame[BACK_IMU_frame["gt"] != 0].copy()
        
        BACK_IMU_frame.loc[:, 'Person'] = user_index
        BACK_IMU_frame.loc[:, 'Position'] = "Back"
        BACK_frame[index] = BACK_IMU_frame.copy()
        BACK_frame[index].reset_index(drop=True, inplace=True)
        
        
        
        
        RUA_IMU_frame = dataframe[[50,51,52,243]].copy()
        RUA_IMU_frame.rename(columns={50: 'Acc_x', 51:'Acc_y', 52: 'Acc_z', 243: 'gt'}, inplace=True)
        RUA_IMU_frame = RUA_IMU_frame.dropna()
        RUA_IMU_frame = RUA_IMU_frame[RUA_IMU_frame["gt"] != 0].copy()
        
        RUA_IMU_frame.loc[:, 'Person'] = user_index
        RUA_IMU_frame.loc[:, 'Position'] = "Right Upper Arm"
        RUA_frame[index] = RUA_IMU_frame.copy()
        RUA_frame[index].reset_index(drop=True, inplace=True)
        
        
        
        RLA_IMU_frame = dataframe[[63,64,65,243]].copy()
        RLA_IMU_frame.rename(columns={63: 'Acc_x', 64:'Acc_y', 65: 'Acc_z', 243: 'gt'}, inplace=True)
        RLA_IMU_frame = RLA_IMU_frame.dropna()
        RLA_IMU_frame = RLA_IMU_frame[RLA_IMU_frame["gt"] != 0].copy()

        RLA_IMU_frame.loc[:, 'Person'] = user_index
        RLA_IMU_frame.loc[:, 'Position'] = "Right Lower Arm"
        RLA_frame[index] = RLA_IMU_frame.copy()
        RLA_frame[index].reset_index(drop=True, inplace=True)
        
        
        
        LUA_IMU_frame = dataframe[[76,77,78,243]].copy()
        LUA_IMU_frame.rename(columns={76: 'Acc_x', 77:'Acc_y', 78: 'Acc_z', 243: 'gt'}, inplace=True)
        LUA_IMU_frame = LUA_IMU_frame.dropna()
        LUA_IMU_frame = LUA_IMU_frame[LUA_IMU_frame["gt"] != 0].copy()

        LUA_IMU_frame.loc[:, 'Person'] = user_index
        LUA_IMU_frame.loc[:, 'Position'] = "Left Upper Arm"
        LUA_frame[index] = LUA_IMU_frame.copy()
        LUA_frame[index].reset_index(drop=True, inplace=True)
        
        
        
        LLA_IMU_frame = dataframe[[89,90,91,243]].copy()
        LLA_IMU_frame.rename(columns={89: 'Acc_x', 90:'Acc_y', 91: 'Acc_z', 243: 'gt'}, inplace=True)
        LLA_IMU_frame = LLA_IMU_frame.dropna()
        LLA_IMU_frame = LLA_IMU_frame[LLA_IMU_frame["gt"] != 0].copy()

        LLA_IMU_frame.loc[:, 'Person'] = user_index
        LLA_IMU_frame.loc[:, 'Position'] = "Left Lower Arm"
        LLA_frame[index] = LLA_IMU_frame.copy()
        LLA_frame[index].reset_index(drop=True, inplace=True)
        


# In[12]:


BACK_frame[15].head()


# In[13]:


BACK_frame[15]['gt'].unique()


# #### Ground truth has to start from 0

# In[14]:


np.unique(BACK_frame[11]['gt'])


# In[15]:


User1_Back_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User1_RUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User1_RLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User1_LUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User1_LLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])

User2_Back_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User2_RUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User2_RLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User2_LUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User2_LLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])

User3_Back_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User3_RUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User3_RLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User3_LUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User3_LLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])

User4_Back_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User4_RUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User4_RLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User4_LUA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
User4_LLA_frame_total = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])


# In[16]:


back_shape = 0
for user_index in range(1,5):
    for run_index in range(1,6):
        index = (user_index * 10) + run_index

        if user_index == 1:
            User1_Back_frame_total = User1_Back_frame_total.append(BACK_frame[index], ignore_index=True)
            User1_RUA_frame_total = User1_RUA_frame_total.append(RUA_frame[index], ignore_index=True)
            User1_RLA_frame_total = User1_RLA_frame_total.append(RLA_frame[index], ignore_index=True)
            User1_LUA_frame_total = User1_LUA_frame_total.append(LUA_frame[index], ignore_index=True)
            User1_LLA_frame_total = User1_LLA_frame_total.append(LLA_frame[index], ignore_index=True)
        elif user_index == 2:
            User2_Back_frame_total = User2_Back_frame_total.append(BACK_frame[index], ignore_index=True)
            User2_RUA_frame_total = User2_RUA_frame_total.append(RUA_frame[index], ignore_index=True)
            User2_RLA_frame_total = User2_RLA_frame_total.append(RLA_frame[index], ignore_index=True)
            User2_LUA_frame_total = User2_LUA_frame_total.append(LUA_frame[index], ignore_index=True)
            User2_LLA_frame_total = User2_LLA_frame_total.append(LLA_frame[index], ignore_index=True)
        elif user_index == 3:
            User3_Back_frame_total = User3_Back_frame_total.append(BACK_frame[index], ignore_index=True)
            User3_RUA_frame_total = User3_RUA_frame_total.append(RUA_frame[index], ignore_index=True)
            User3_RLA_frame_total = User3_RLA_frame_total.append(RLA_frame[index], ignore_index=True)
            User3_LUA_frame_total = User3_LUA_frame_total.append(LUA_frame[index], ignore_index=True)
            User3_LLA_frame_total = User3_LLA_frame_total.append(LLA_frame[index], ignore_index=True)
        elif user_index == 4:
            User4_Back_frame_total = User4_Back_frame_total.append(BACK_frame[index], ignore_index=True)
            User4_RUA_frame_total = User4_RUA_frame_total.append(RUA_frame[index], ignore_index=True)
            User4_RLA_frame_total = User4_RLA_frame_total.append(RLA_frame[index], ignore_index=True)
            User4_LUA_frame_total = User4_LUA_frame_total.append(LUA_frame[index], ignore_index=True)
            User4_LLA_frame_total = User4_LLA_frame_total.append(LLA_frame[index], ignore_index=True)


# In[17]:


np.unique(User1_Back_frame_total['gt'])


# #### User 1 - Label Mapping

# In[18]:


User1_Back_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User1_Back_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User1_RUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User1_RUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User1_RLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User1_RLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User1_LUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User1_LUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User1_LLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User1_LLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)


# In[19]:


np.unique(User1_LLA_frame_total['gt'])


# #### User 2 - Label Mapping

# In[20]:


User2_Back_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User2_Back_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User2_RUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User2_RUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User2_RLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User2_RLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User2_LUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User2_LUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User2_LLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User2_LLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)


# #### User 3 - Label Mapping

# In[21]:


User3_Back_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User3_Back_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User3_RUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User3_RUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User3_RLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User3_RLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User3_LUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User3_LUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User3_LLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User3_LLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)


# #### User 4 - Label Mapping

# In[22]:


User4_Back_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User4_Back_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User4_RUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User4_RUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User4_RLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User4_RLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User4_LUA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User4_LUA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)

User4_LLA_frame_total.replace({'gt' : { 1:100, 2:200, 4:400, 5:500}}, inplace = True)
User4_LLA_frame_total.replace({'gt' : { 100:1, 200:3, 400:0, 500:2}}, inplace = True)


# ### User Specific, Position-wise Standardization and Normalization

# #### User 1 - Back

# In[23]:


person_gt = np.array(User1_Back_frame_total['Person'])
gt = np.array(User1_Back_frame_total['gt'])
position_gt = np.array(User1_Back_frame_total['Position'])

User1_Back_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User1_Back_frame_total.columns

np_scaled = standardization_and_normalization(User1_Back_frame_total)
User1_Back_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User1_Back_frame_total["Person"] = person_gt
User1_Back_frame_total["gt"] = gt
User1_Back_frame_total["Position"] = position_gt


# In[24]:


np.unique(User1_Back_frame_total['gt'])


# In[25]:


np.unique(User1_Back_frame_total['Person'])


# In[26]:


np.unique(User1_Back_frame_total['Position'])


# #### User 2 - Back

# In[27]:


person_gt = np.array(User2_Back_frame_total['Person'])
gt = np.array(User2_Back_frame_total['gt'])
position_gt = np.array(User2_Back_frame_total['Position'])

User2_Back_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User2_Back_frame_total.columns

np_scaled = standardization_and_normalization(User2_Back_frame_total)
User2_Back_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User2_Back_frame_total["Person"] = person_gt
User2_Back_frame_total["gt"] = gt
User2_Back_frame_total["Position"] = position_gt


# #### User 3 - Back

# In[28]:


person_gt = np.array(User3_Back_frame_total['Person'])
gt = np.array(User3_Back_frame_total['gt'])
position_gt = np.array(User3_Back_frame_total['Position'])

User3_Back_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User3_Back_frame_total.columns

np_scaled = standardization_and_normalization(User3_Back_frame_total)
User3_Back_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User3_Back_frame_total["Person"] = person_gt
User3_Back_frame_total["gt"] = gt
User3_Back_frame_total["Position"] = position_gt


# #### User 4 - Back

# In[29]:


person_gt = np.array(User4_Back_frame_total['Person'])
gt = np.array(User4_Back_frame_total['gt'])
position_gt = np.array(User4_Back_frame_total['Position'])

User4_Back_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User4_Back_frame_total.columns

np_scaled = standardization_and_normalization(User4_Back_frame_total)
User4_Back_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User4_Back_frame_total["Person"] = person_gt
User4_Back_frame_total["gt"] = gt
User4_Back_frame_total["Position"] = position_gt


# In[30]:


User4_Back_frame_total.head()


# In[31]:


np.max(User4_Back_frame_total['Acc_x'])


# #### User1-RUA

# In[32]:


person_gt = np.array(User1_RUA_frame_total['Person'])
gt = np.array(User1_RUA_frame_total['gt'])
position_gt = np.array(User1_RUA_frame_total['Position'])

User1_RUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User1_RUA_frame_total.columns

np_scaled = standardization_and_normalization(User1_RUA_frame_total)
User1_RUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User1_RUA_frame_total["Person"] = person_gt
User1_RUA_frame_total["gt"] = gt
User1_RUA_frame_total["Position"] = position_gt


# #### User2 - RUA

# In[33]:


person_gt = np.array(User2_RUA_frame_total['Person'])
gt = np.array(User2_RUA_frame_total['gt'])
position_gt = np.array(User2_RUA_frame_total['Position'])

User2_RUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User2_RUA_frame_total.columns

np_scaled = standardization_and_normalization(User2_RUA_frame_total)
User2_RUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User2_RUA_frame_total["Person"] = person_gt
User2_RUA_frame_total["gt"] = gt
User2_RUA_frame_total["Position"] = position_gt


# #### User3 - RUA

# In[34]:


person_gt = np.array(User3_RUA_frame_total['Person'])
gt = np.array(User3_RUA_frame_total['gt'])
position_gt = np.array(User3_RUA_frame_total['Position'])

User3_RUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User3_RUA_frame_total.columns

np_scaled = standardization_and_normalization(User3_RUA_frame_total)
User3_RUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User3_RUA_frame_total["Person"] = person_gt
User3_RUA_frame_total["gt"] = gt
User3_RUA_frame_total["Position"] = position_gt


# #### USer4 - RUA

# In[35]:


person_gt = np.array(User4_RUA_frame_total['Person'])
gt = np.array(User4_RUA_frame_total['gt'])
position_gt = np.array(User4_RUA_frame_total['Position'])

User4_RUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User4_RUA_frame_total.columns

np_scaled = standardization_and_normalization(User4_RUA_frame_total)
User4_RUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User4_RUA_frame_total["Person"] = person_gt
User4_RUA_frame_total["gt"] = gt
User4_RUA_frame_total["Position"] = position_gt


# #### User 1 - RLA

# In[36]:


person_gt = np.array(User1_RLA_frame_total['Person'])
gt = np.array(User1_RLA_frame_total['gt'])
position_gt = np.array(User1_RLA_frame_total['Position'])

User1_RLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User1_RLA_frame_total.columns

np_scaled = standardization_and_normalization(User1_RLA_frame_total)
User1_RLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User1_RLA_frame_total["Person"] = person_gt
User1_RLA_frame_total["gt"] = gt
User1_RLA_frame_total["Position"] = position_gt


# #### User 2 - RLA

# In[37]:


person_gt = np.array(User2_RLA_frame_total['Person'])
gt = np.array(User2_RLA_frame_total['gt'])
position_gt = np.array(User2_RLA_frame_total['Position'])

User2_RLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User2_RLA_frame_total.columns

np_scaled = standardization_and_normalization(User2_RLA_frame_total)
User2_RLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User2_RLA_frame_total["Person"] = person_gt
User2_RLA_frame_total["gt"] = gt
User2_RLA_frame_total["Position"] = position_gt


# #### User 3 - RLA

# In[38]:


person_gt = np.array(User3_RLA_frame_total['Person'])
gt = np.array(User3_RLA_frame_total['gt'])
position_gt = np.array(User3_RLA_frame_total['Position'])

User3_RLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User3_RLA_frame_total.columns

np_scaled = standardization_and_normalization(User3_RLA_frame_total)
User3_RLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User3_RLA_frame_total["Person"] = person_gt
User3_RLA_frame_total["gt"] = gt
User3_RLA_frame_total["Position"] = position_gt


# #### User 4 - RLA

# In[39]:


person_gt = np.array(User4_RLA_frame_total['Person'])
gt = np.array(User4_RLA_frame_total['gt'])
position_gt = np.array(User4_RLA_frame_total['Position'])

User4_RLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User4_RLA_frame_total.columns

np_scaled = standardization_and_normalization(User4_RLA_frame_total)
User4_RLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User4_RLA_frame_total["Person"] = person_gt
User4_RLA_frame_total["gt"] = gt
User4_RLA_frame_total["Position"] = position_gt


# #### User 1 - LUA

# In[40]:


person_gt = np.array(User1_LUA_frame_total['Person'])
gt = np.array(User1_LUA_frame_total['gt'])
position_gt = np.array(User1_LUA_frame_total['Position'])

User1_LUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User1_LUA_frame_total.columns

np_scaled = standardization_and_normalization(User1_LUA_frame_total)
User1_LUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User1_LUA_frame_total["Person"] = person_gt
User1_LUA_frame_total["gt"] = gt
User1_LUA_frame_total["Position"] = position_gt


# #### User 2 - LUA

# In[41]:


person_gt = np.array(User2_LUA_frame_total['Person'])
gt = np.array(User2_LUA_frame_total['gt'])
position_gt = np.array(User2_LUA_frame_total['Position'])

User2_LUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User2_LUA_frame_total.columns

np_scaled = standardization_and_normalization(User2_LUA_frame_total)
User2_LUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User2_LUA_frame_total["Person"] = person_gt
User2_LUA_frame_total["gt"] = gt
User2_LUA_frame_total["Position"] = position_gt


# #### User 3 - LUA

# In[42]:


person_gt = np.array(User3_LUA_frame_total['Person'])
gt = np.array(User3_LUA_frame_total['gt'])
position_gt = np.array(User3_LUA_frame_total['Position'])

User3_LUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User3_LUA_frame_total.columns

np_scaled = standardization_and_normalization(User3_LUA_frame_total)
User3_LUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User3_LUA_frame_total["Person"] = person_gt
User3_LUA_frame_total["gt"] = gt
User3_LUA_frame_total["Position"] = position_gt


# #### User 4 - LUA

# In[43]:


person_gt = np.array(User4_LUA_frame_total['Person'])
gt = np.array(User4_LUA_frame_total['gt'])
position_gt = np.array(User4_LUA_frame_total['Position'])

User4_LUA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User4_LUA_frame_total.columns

np_scaled = standardization_and_normalization(User4_LUA_frame_total)
User4_LUA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User4_LUA_frame_total["Person"] = person_gt
User4_LUA_frame_total["gt"] = gt
User4_LUA_frame_total["Position"] = position_gt


# #### User 1 - LLA

# In[44]:


person_gt = np.array(User1_LLA_frame_total['Person'])
gt = np.array(User1_LLA_frame_total['gt'])
position_gt = np.array(User1_LLA_frame_total['Position'])

User1_LLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User1_LLA_frame_total.columns

np_scaled = standardization_and_normalization(User1_LLA_frame_total)
User1_LLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User1_LLA_frame_total["Person"] = person_gt
User1_LLA_frame_total["gt"] = gt
User1_LLA_frame_total["Position"] = position_gt


# #### User 2 - LLA

# In[45]:


person_gt = np.array(User2_LLA_frame_total['Person'])
gt = np.array(User2_LLA_frame_total['gt'])
position_gt = np.array(User2_LLA_frame_total['Position'])

User2_LLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User2_LLA_frame_total.columns

np_scaled = standardization_and_normalization(User2_LLA_frame_total)
User2_LLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User2_LLA_frame_total["Person"] = person_gt
User2_LLA_frame_total["gt"] = gt
User2_LLA_frame_total["Position"] = position_gt


# #### User 3 - LLA

# In[46]:


person_gt = np.array(User3_LLA_frame_total['Person'])
gt = np.array(User3_LLA_frame_total['gt'])
position_gt = np.array(User3_LLA_frame_total['Position'])

User3_LLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User3_LLA_frame_total.columns

np_scaled = standardization_and_normalization(User3_LLA_frame_total)
User3_LLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User3_LLA_frame_total["Person"] = person_gt
User3_LLA_frame_total["gt"] = gt
User3_LLA_frame_total["Position"] = position_gt


# #### User 4 - LLA

# In[47]:


person_gt = np.array(User4_LLA_frame_total['Person'])
gt = np.array(User4_LLA_frame_total['gt'])
position_gt = np.array(User4_LLA_frame_total['Position'])

User4_LLA_frame_total.drop(['Person','gt','Position'], axis=1, inplace=True)
column_name = User4_LLA_frame_total.columns

np_scaled = standardization_and_normalization(User4_LLA_frame_total)
User4_LLA_frame_total = pd.DataFrame(np_scaled, columns=column_name)

User4_LLA_frame_total["Person"] = person_gt
User4_LLA_frame_total["gt"] = gt
User4_LLA_frame_total["Position"] = position_gt


# In[48]:


User1_LLA_frame_total.head()


# In[49]:


User1_LLA_frame_total.tail()


# ## Train-Valid-Test Split

# #### User 1

# In[50]:


activity_set = np.unique(User1_Back_frame_total['gt'])
total_activity = len(np.unique(User1_Back_frame_total['gt']))


# In[51]:


total_activity


# In[52]:


activity_set


# In[53]:


user1_back_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_back_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_back_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    
    activity_frame = User1_Back_frame_total[User1_Back_frame_total['gt'] == activity_set[activity_index]]

    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user1_back_train = user1_back_train.append(train, ignore_index=True, sort = False)
    user1_back_valid = user1_back_valid.append(valid, ignore_index=True, sort = False)
    user1_back_test = user1_back_test.append(test, ignore_index=True, sort = False)
    
user1_back_train.to_csv (save_path+'user1_back_train'+'.csv', index = None, header=True)
user1_back_valid.to_csv (save_path+'user1_back_valid'+'.csv', index = None, header=True)
user1_back_test.to_csv (save_path+'user1_back_test'+'.csv', index = None, header=True)


# plot_activity(user1_back_train, "Default")
# plot_activity(user1_back_valid, "Default")


# In[54]:


activity_set = np.unique(User1_RUA_frame_total['gt'])
total_activity = len(np.unique(User1_RUA_frame_total['gt']))


# In[55]:


total_activity


# In[56]:


user1_RUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_RUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_RUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User1_RUA_frame_total[User1_RUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user1_RUA_train = user1_RUA_train.append(train, ignore_index=True, sort = False)
    user1_RUA_valid = user1_RUA_valid.append(valid, ignore_index=True, sort = False)
    user1_RUA_test = user1_RUA_test.append(test, ignore_index=True, sort = False)
    
user1_RUA_train.to_csv (save_path+'user1_RUA_train'+'.csv', index = None, header=True)
user1_RUA_valid.to_csv (save_path+'user1_RUA_valid'+'.csv', index = None, header=True)
user1_RUA_test.to_csv (save_path+'user1_RUA_test'+'.csv', index = None, header=True)


# In[57]:


activity_set = np.unique(User1_RLA_frame_total['gt'])
total_activity = len(np.unique(User1_RLA_frame_total['gt']))


# In[58]:


total_activity


# In[59]:


user1_RLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_RLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_RLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User1_RLA_frame_total[User1_RLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user1_RLA_train = user1_RLA_train.append(train, ignore_index=True, sort = False)
    user1_RLA_valid = user1_RLA_valid.append(valid, ignore_index=True, sort = False)
    user1_RLA_test = user1_RLA_test.append(test, ignore_index=True, sort = False)
    
user1_RLA_train.to_csv (save_path+'user1_RLA_train'+'.csv', index = None, header=True)
user1_RLA_valid.to_csv (save_path+'user1_RLA_valid'+'.csv', index = None, header=True)
user1_RLA_test.to_csv (save_path+'user1_RLA_test'+'.csv', index = None, header=True)


# In[60]:


activity_set = np.unique(User1_LUA_frame_total['gt'])
total_activity = len(np.unique(User1_LUA_frame_total['gt']))


# In[61]:


total_activity


# In[62]:


user1_LUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_LUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_LUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User1_LUA_frame_total[User1_LUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user1_LUA_train = user1_LUA_train.append(train, ignore_index=True, sort = False)
    user1_LUA_valid = user1_LUA_valid.append(valid, ignore_index=True, sort = False)
    user1_LUA_test = user1_LUA_test.append(test, ignore_index=True, sort = False)
    
user1_LUA_train.to_csv (save_path+'user1_LUA_train'+'.csv', index = None, header=True)
user1_LUA_valid.to_csv (save_path+'user1_LUA_valid'+'.csv', index = None, header=True)
user1_LUA_test.to_csv (save_path+'user1_LUA_test'+'.csv', index = None, header=True)


# In[63]:


activity_set = np.unique(User1_LLA_frame_total['gt'])
total_activity = len(np.unique(User1_LLA_frame_total['gt']))


# In[64]:


total_activity


# In[65]:


user1_LLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_LLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user1_LLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User1_LLA_frame_total[User1_LLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user1_LLA_train = user1_LLA_train.append(train, ignore_index=True, sort = False)
    user1_LLA_valid = user1_LLA_valid.append(valid, ignore_index=True, sort = False)
    user1_LLA_test = user1_LLA_test.append(test, ignore_index=True, sort = False)
    
user1_LLA_train.to_csv (save_path+'user1_LLA_train'+'.csv', index = None, header=True)
user1_LLA_valid.to_csv (save_path+'user1_LLA_valid'+'.csv', index = None, header=True)
user1_LLA_test.to_csv (save_path+'user1_LLA_test'+'.csv', index = None, header=True)


# #### User 2

# In[66]:


activity_set = np.unique(User2_Back_frame_total['gt'])
total_activity = len(np.unique(User2_Back_frame_total['gt']))


# In[67]:


total_activity


# In[68]:


user2_back_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_back_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_back_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User2_Back_frame_total[User2_Back_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user2_back_train = user2_back_train.append(train, ignore_index=True, sort = False)
    user2_back_valid = user2_back_valid.append(valid, ignore_index=True, sort = False)
    user2_back_test = user2_back_test.append(test, ignore_index=True, sort = False)
    
    
user2_back_train.to_csv (save_path+'user2_back_train'+'.csv', index = None, header=True)
user2_back_valid.to_csv (save_path+'user2_back_valid'+'.csv', index = None, header=True)
user2_back_test.to_csv (save_path+'user2_back_test'+'.csv', index = None, header=True)


# In[69]:


activity_set = np.unique(User2_RUA_frame_total['gt'])
total_activity = len(np.unique(User2_RUA_frame_total['gt']))


# In[70]:


total_activity


# In[71]:


user2_RUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_RUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_RUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User2_RUA_frame_total[User2_RUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user2_RUA_train = user2_RUA_train.append(train, ignore_index=True, sort = False)
    user2_RUA_valid = user2_RUA_valid.append(valid, ignore_index=True, sort = False)
    user2_RUA_test = user2_RUA_test.append(test, ignore_index=True, sort = False)
    
user2_RUA_train.to_csv (save_path+'user2_RUA_train'+'.csv', index = None, header=True)
user2_RUA_valid.to_csv (save_path+'user2_RUA_valid'+'.csv', index = None, header=True)
user2_RUA_test.to_csv (save_path+'user2_RUA_test'+'.csv', index = None, header=True)


# In[72]:


activity_set = np.unique(User2_RLA_frame_total['gt'])
total_activity = len(np.unique(User2_RLA_frame_total['gt']))


# In[73]:


total_activity


# In[74]:


user2_RLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_RLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_RLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User2_RLA_frame_total[User2_RLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user2_RLA_train = user2_RLA_train.append(train, ignore_index=True, sort = False)
    user2_RLA_valid = user2_RLA_valid.append(valid, ignore_index=True, sort = False)
    user2_RLA_test = user2_RLA_test.append(test, ignore_index=True, sort = False)
    
user2_RLA_train.to_csv (save_path+'user2_RLA_train'+'.csv', index = None, header=True)
user2_RLA_valid.to_csv (save_path+'user2_RLA_valid'+'.csv', index = None, header=True)
user2_RLA_test.to_csv (save_path+'user2_RLA_test'+'.csv', index = None, header=True)


# In[75]:


activity_set = np.unique(User2_LUA_frame_total['gt'])
total_activity = len(np.unique(User2_LUA_frame_total['gt']))


# In[76]:


total_activity


# In[77]:


user2_LUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_LUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_LUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User2_LUA_frame_total[User2_LUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user2_LUA_train = user2_LUA_train.append(train, ignore_index=True, sort = False)
    user2_LUA_valid = user2_LUA_valid.append(valid, ignore_index=True, sort = False)
    user2_LUA_test = user2_LUA_test.append(test, ignore_index=True, sort = False)
    
user2_LUA_train.to_csv (save_path+'user2_LUA_train'+'.csv', index = None, header=True)
user2_LUA_valid.to_csv (save_path+'user2_LUA_valid'+'.csv', index = None, header=True)
user2_LUA_test.to_csv (save_path+'user2_LUA_test'+'.csv', index = None, header=True)


# In[78]:


activity_set = np.unique(User2_LLA_frame_total['gt'])
total_activity = len(np.unique(User2_LLA_frame_total['gt']))


# In[79]:


total_activity


# In[80]:


user2_LLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_LLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user2_LLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User2_LLA_frame_total[User2_LLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user2_LLA_train = user2_LLA_train.append(train, ignore_index=True, sort = False)
    user2_LLA_valid = user2_LLA_valid.append(valid, ignore_index=True, sort = False)
    user2_LLA_test = user2_LLA_test.append(test, ignore_index=True, sort = False)
    
user2_LLA_train.to_csv (save_path+'user2_LLA_train'+'.csv', index = None, header=True)
user2_LLA_valid.to_csv (save_path+'user2_LLA_valid'+'.csv', index = None, header=True)
user2_LLA_test.to_csv (save_path+'user2_LLA_test'+'.csv', index = None, header=True)


# #### User 3

# In[81]:


activity_set = np.unique(User3_Back_frame_total['gt'])
total_activity = len(np.unique(User3_Back_frame_total['gt']))


# In[82]:


total_activity


# In[83]:


user3_back_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_back_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_back_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User3_Back_frame_total[User3_Back_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user3_back_train = user3_back_train.append(train, ignore_index=True, sort = False)
    user3_back_valid = user3_back_valid.append(valid, ignore_index=True, sort = False)
    user3_back_test = user3_back_test.append(test, ignore_index=True, sort = False)

user3_back_train.to_csv (save_path+'user3_back_train'+'.csv', index = None, header=True)
user3_back_valid.to_csv (save_path+'user3_back_valid'+'.csv', index = None, header=True)
user3_back_test.to_csv (save_path+'user3_back_test'+'.csv', index = None, header=True)


# In[84]:


activity_set = np.unique(User3_RUA_frame_total['gt'])
total_activity = len(np.unique(User3_RUA_frame_total['gt']))


# In[85]:


total_activity


# In[86]:


user3_RUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_RUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_RUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User3_RUA_frame_total[User3_RUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user3_RUA_train = user3_RUA_train.append(train, ignore_index=True, sort = False)
    user3_RUA_valid = user3_RUA_valid.append(valid, ignore_index=True, sort = False)
    user3_RUA_test = user3_RUA_test.append(test, ignore_index=True, sort = False)
    
user3_RUA_train.to_csv (save_path+'user3_RUA_train'+'.csv', index = None, header=True)
user3_RUA_valid.to_csv (save_path+'user3_RUA_valid'+'.csv', index = None, header=True)
user3_RUA_test.to_csv (save_path+'user3_RUA_test'+'.csv', index = None, header=True)


# In[87]:


activity_set = np.unique(User3_RLA_frame_total['gt'])
total_activity = len(np.unique(User3_RLA_frame_total['gt']))


# In[88]:


total_activity


# In[89]:


user3_RLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_RLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_RLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User3_RLA_frame_total[User3_RLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user3_RLA_train = user3_RLA_train.append(train, ignore_index=True, sort = False)
    user3_RLA_valid = user3_RLA_valid.append(valid, ignore_index=True, sort = False)
    user3_RLA_test = user3_RLA_test.append(test, ignore_index=True, sort = False)
    
user3_RLA_train.to_csv (save_path+'user3_RLA_train'+'.csv', index = None, header=True)
user3_RLA_valid.to_csv (save_path+'user3_RLA_valid'+'.csv', index = None, header=True)
user3_RLA_test.to_csv (save_path+'user3_RLA_test'+'.csv', index = None, header=True)


# In[90]:


activity_set = np.unique(User3_LUA_frame_total['gt'])
total_activity = len(np.unique(User3_LUA_frame_total['gt']))


# In[91]:


total_activity


# In[92]:


user3_LUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_LUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_LUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User3_LUA_frame_total[User3_LUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user3_LUA_train = user3_LUA_train.append(train, ignore_index=True, sort = False)
    user3_LUA_valid = user3_LUA_valid.append(valid, ignore_index=True, sort = False)
    user3_LUA_test = user3_LUA_test.append(test, ignore_index=True, sort = False)
    
user3_LUA_train.to_csv (save_path+'user3_LUA_train'+'.csv', index = None, header=True)
user3_LUA_valid.to_csv (save_path+'user3_LUA_valid'+'.csv', index = None, header=True)
user3_LUA_test.to_csv (save_path+'user3_LUA_test'+'.csv', index = None, header=True)


# In[93]:


activity_set = np.unique(User3_LLA_frame_total['gt'])
total_activity = len(np.unique(User3_LLA_frame_total['gt']))


# In[94]:


total_activity


# In[95]:


user3_LLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_LLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user3_LLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User3_LLA_frame_total[User3_LLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user3_LLA_train = user3_LLA_train.append(train, ignore_index=True, sort = False)
    user3_LLA_valid = user3_LLA_valid.append(valid, ignore_index=True, sort = False)
    user3_LLA_test = user3_LLA_test.append(test, ignore_index=True, sort = False)
    
user3_LLA_train.to_csv (save_path+'user3_LLA_train'+'.csv', index = None, header=True)
user3_LLA_valid.to_csv (save_path+'user3_LLA_valid'+'.csv', index = None, header=True)
user3_LLA_test.to_csv (save_path+'user3_LLA_test'+'.csv', index = None, header=True)


# #### User 4

# In[96]:


activity_set = np.unique(User4_Back_frame_total['gt'])
total_activity = len(np.unique(User4_Back_frame_total['gt']))


# In[97]:


total_activity


# In[98]:


user4_back_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_back_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_back_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User4_Back_frame_total[User4_Back_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user4_back_train = user4_back_train.append(train, ignore_index=True, sort = False)
    user4_back_valid = user4_back_valid.append(valid, ignore_index=True, sort = False)
    user4_back_test = user4_back_test.append(test, ignore_index=True, sort = False)
    
user4_back_train.to_csv (save_path+'user4_back_train'+'.csv', index = None, header=True)
user4_back_valid.to_csv (save_path+'user4_back_valid'+'.csv', index = None, header=True)
user4_back_test.to_csv (save_path+'user4_back_test'+'.csv', index = None, header=True)


# In[99]:


activity_set = np.unique(User4_RUA_frame_total['gt'])
total_activity = len(np.unique(User4_RUA_frame_total['gt']))


# In[100]:


total_activity


# In[101]:


user4_RUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_RUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_RUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User4_RUA_frame_total[User4_RUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user4_RUA_train = user4_RUA_train.append(train, ignore_index=True, sort = False)
    user4_RUA_valid = user4_RUA_valid.append(valid, ignore_index=True, sort = False)
    user4_RUA_test = user4_RUA_test.append(test, ignore_index=True, sort = False)
    
user4_RUA_train.to_csv (save_path+'user4_RUA_train'+'.csv', index = None, header=True)
user4_RUA_valid.to_csv (save_path+'user4_RUA_valid'+'.csv', index = None, header=True)
user4_RUA_test.to_csv (save_path+'user4_RUA_test'+'.csv', index = None, header=True)


# In[102]:


activity_set = np.unique(User4_RLA_frame_total['gt'])
total_activity = len(np.unique(User4_RLA_frame_total['gt']))


# In[103]:


total_activity


# In[104]:


user4_RLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_RLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_RLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User4_RLA_frame_total[User4_RLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user4_RLA_train = user4_RLA_train.append(train, ignore_index=True, sort = False)
    user4_RLA_valid = user4_RLA_valid.append(valid, ignore_index=True, sort = False)
    user4_RLA_test = user4_RLA_test.append(test, ignore_index=True, sort = False)
    
user4_RLA_train.to_csv (save_path+'user4_RLA_train'+'.csv', index = None, header=True)
user4_RLA_valid.to_csv (save_path+'user4_RLA_valid'+'.csv', index = None, header=True)
user4_RLA_test.to_csv (save_path+'user4_RLA_test'+'.csv', index = None, header=True)


# In[105]:


activity_set = np.unique(User4_LUA_frame_total['gt'])
total_activity = len(np.unique(User4_LUA_frame_total['gt']))


# In[106]:


total_activity


# In[107]:


user4_LUA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_LUA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_LUA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User4_LUA_frame_total[User4_LUA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user4_LUA_train = user4_LUA_train.append(train, ignore_index=True, sort = False)
    user4_LUA_valid = user4_LUA_valid.append(valid, ignore_index=True, sort = False)
    user4_LUA_test = user4_LUA_test.append(test, ignore_index=True, sort = False)
    
user4_LUA_train.to_csv (save_path+'user4_LUA_train'+'.csv', index = None, header=True)
user4_LUA_valid.to_csv (save_path+'user4_LUA_valid'+'.csv', index = None, header=True)
user4_LUA_test.to_csv (save_path+'user4_LUA_test'+'.csv', index = None, header=True)


# In[108]:


activity_set = np.unique(User4_LLA_frame_total['gt'])
total_activity = len(np.unique(User4_LLA_frame_total['gt']))


# In[109]:


total_activity


# In[110]:


user4_LLA_train = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_LLA_valid = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
user4_LLA_test = pd.DataFrame(columns=['Acc_x', 'Acc_y', 'Acc_z', 'gt', 'Person', 'Position'])
for activity_index in range(0, total_activity):
    activity_frame = User4_LLA_frame_total[User4_LLA_frame_total['gt'] == activity_set[activity_index]]
    
    train_upto = int(activity_frame.shape[0]*train_percentage)    
    valid_upto = int(activity_frame.shape[0]*valid_percentage)  
        
    train = activity_frame[0:train_upto].copy()
    valid = activity_frame[train_upto+1 : train_upto+valid_upto].copy()
    test = activity_frame[train_upto+valid_upto+1 : activity_frame.shape[0]].copy()
    
    user4_LLA_train = user4_LLA_train.append(train, ignore_index=True, sort = False)
    user4_LLA_valid = user4_LLA_valid.append(valid, ignore_index=True, sort = False)
    user4_LLA_test = user4_LLA_test.append(test, ignore_index=True, sort = False)

user4_LLA_train.to_csv (save_path+'user4_LLA_train'+'.csv', index = None, header=True)
user4_LLA_valid.to_csv (save_path+'user4_LLA_valid'+'.csv', index = None, header=True)
user4_LLA_test.to_csv (save_path+'user4_LLA_test'+'.csv', index = None, header=True)


# In[111]:


np.unique(user4_LLA_train['gt'])


# In[112]:


np.unique(user4_LLA_valid['gt'])


# In[113]:


len(user4_LLA_valid)


# ### Code block for merging all user train and test data for a specific position

# ## Prepare Final Dataset

# In[114]:


BACK_dataset_train = []
RUA_dataset_train = []
RLA_dataset_train = []
LUA_dataset_train = []
LLA_dataset_train = []
BACK_gt_train = []
RUA_gt_train = []
RLA_gt_train = []
LUA_gt_train = []
LLA_gt_train = []


# In[115]:


BACK_dataset_valid = []
RUA_dataset_valid = []
RLA_dataset_valid = []
LUA_dataset_valid = []
LLA_dataset_valid = []
BACK_gt_valid = []
RUA_gt_valid = []
RLA_gt_valid = []
LUA_gt_valid = []
LLA_gt_valid = []


# In[116]:


BACK_dataset_test = []
RUA_dataset_test = []
RLA_dataset_test = []
LUA_dataset_test = []
LLA_dataset_test = []
BACK_gt_test = []
RUA_gt_test = []
RLA_gt_test = []
LUA_gt_test = []
LLA_gt_test = []


# ### Code Block for taking position-wise all user train and validation data

# In[117]:


for position_index in tqdm.tqdm(range(0,5)): #Back, RUA, RLA, LUA, LLA
    for split_index in range(0,3):
        file_name = user[selected_user-1] + "_" + position[position_index]+'_'+item[split_index]
        
        print(file_name)
        df = pd.read_csv(save_path+file_name+'.csv', sep=",")   
        len_df = df.shape[0]
        narray = df.to_numpy()

        for i in range(0, len_df, step_size):
            window = narray[i:i+win_size, FROM:TO]
            
            if window.shape[0] != win_size:
                continue
            else:
                reshaped_window = window.reshape(1,win_size,1,AXIS)
                gt = np.bincount(narray[i:i+win_size,START:END].astype(int).ravel()).argmax()
                
                if position_index == 0:
                    if split_index == 0:
                        BACK_dataset_train.append(reshaped_window)
                        BACK_gt_train.append(gt)
                    elif split_index == 1:
                        BACK_dataset_valid.append(reshaped_window)
                        BACK_gt_valid.append(gt)
                    elif split_index == 2:
                        BACK_dataset_test.append(reshaped_window)
                        BACK_gt_test.append(gt)
                elif position_index == 1:
                    if split_index == 0:
                        RUA_dataset_train.append(reshaped_window)
                        RUA_gt_train.append(gt)
                    elif split_index == 1:
                        RUA_dataset_valid.append(reshaped_window)
                        RUA_gt_valid.append(gt)
                    elif split_index == 2:
                        RUA_dataset_test.append(reshaped_window)
                        RUA_gt_test.append(gt)
                elif position_index == 2:
                    if split_index == 0:
                        RLA_dataset_train.append(reshaped_window)
                        RLA_gt_train.append(gt)
                    elif split_index == 1:
                        RLA_dataset_valid.append(reshaped_window)
                        RLA_gt_valid.append(gt)
                    elif split_index == 2:
                        RLA_dataset_test.append(reshaped_window)
                        RLA_gt_test.append(gt)
                elif position_index == 3:
                    if split_index == 0:
                        LUA_dataset_train.append(reshaped_window)
                        LUA_gt_train.append(gt)
                    elif split_index == 1:
                        LUA_dataset_valid.append(reshaped_window)
                        LUA_gt_valid.append(gt)
                    elif split_index == 2:
                        LUA_dataset_test.append(reshaped_window)
                        LUA_gt_test.append(gt)
                elif position_index == 4:
                    if split_index == 0:
                        LLA_dataset_train.append(reshaped_window)
                        LLA_gt_train.append(gt)
                    elif split_index == 1:
                        LLA_dataset_valid.append(reshaped_window)
                        LLA_gt_valid.append(gt)
                    elif split_index == 2:
                        LLA_dataset_test.append(reshaped_window)
                        LLA_gt_test.append(gt)

