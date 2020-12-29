#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import itertools

def new_accuracy_plot(user=1, save_dir='', final_s1_accuracy=0, final_s2_accuracy=0, final_tar_accuracy=0):
    data = np.loadtxt(save_dir+'msada_result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='Target Validation Accuracy')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='Source1 Validation Accuracy')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='green', label='Source2 Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    
    plot_title = 'User: '+str(user) + "Source1 Acc: "+ str(final_s1_accuracy) + "Source2 Acc: "+ str(final_s2_accuracy) + " Target Acc: "+ str(final_tar_accuracy)
    plt.title(plot_title, fontsize=10)
    plt.savefig(save_dir+plot_title+'.png')
    
    
def new_loss_plot(selected_user=1, save_dir=''):
    label = ['Confusion Loss', 'Discriminator Loss', 'Classification Loss']
    
    data = np.loadtxt(save_dir+'source1_result.csv', delimiter=',')  
    for i in range(3):
        plt.figure()
        plt.plot(range(1, len(data[:, i]) + 1), data[:, i], color='blue', label=label[i])
        plt.legend()
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss (%)', fontsize=14)
        plot_title = "User: "+ str(selected_user)+ ' Source A '+ label[i]
        plt.title(plot_title, fontsize=10)
        image_name = plot_title+'.png'
        plt.savefig(save_dir+image_name)
    
    data = np.loadtxt(save_dir+'source2_result.csv', delimiter=',')
    for i in range(3):
        plt.figure()
        plt.plot(range(1, len(data[:, i]) + 1), data[:, i], color='blue', label=label[i])
        plt.legend()
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss (%)', fontsize=14)
        plot_title = "User: "+ str(selected_user)+' Source B '+ label[i]
        plt.title(plot_title, fontsize=10)
        image_name = plot_title+'.png'
        plt.savefig(save_dir+image_name)

def plot_training_data_distribution_multi_source(s1, s2, target, activity_list, save_dir):
    """
    S1, S2, Target are the windowed dataframe
    Counts the number of samples for each class label and 
    generates the data distribution plot and saves to the provided directory
    """
    
    unique, counts = np.unique(target, return_counts=True)
    target_dict = dict(zip(unique, counts))
    target_samples = np.fromiter(target_dict.values(), dtype=float).astype(int)

    unique, counts = np.unique(s1, return_counts=True)
    s1_dict = dict(zip(unique, counts))
    s1_samples = np.fromiter(s1_dict.values(), dtype=float).astype(int)

    unique, counts = np.unique(s2, return_counts=True)
    s2_dict = dict(zip(unique, counts))
    s2_samples = np.fromiter(s2_dict.values(), dtype=float).astype(int)

    n_groups = len(activity_list)

    plt.figure(figsize=(12,8))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, s1_samples, bar_width, alpha=opacity, color='b', label='Source1')
    rects2 = plt.bar(index + bar_width, s2_samples, bar_width, alpha=opacity, color='g', label='Source2')
    rects3 = plt.bar(index + 2*bar_width, target_samples, bar_width, alpha=opacity, color='m',label='Target')

    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Window Number', fontsize=14)
    plt.title('Window Distribution', fontsize=10)
    plt.xticks(index + bar_width, activity_list, rotation = 45)
    plt.legend()
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir+"Activity Distribution.png")
    
    plt.show()
     
        
        
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', plot_save_path='', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Saves the plot to the provided directory
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center", fontsize=20, 
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(plot_save_path + title + '.png')
    
    
def plot(plot_file_url, name, window, overlap, selected_user, test_accuracy, plot_common_title, plot_save_path):
    if name == 'A':
        data = np.loadtxt(plot_file_url+'/A_result.csv', delimiter=',')
    elif name == 'B':
        data = np.loadtxt(plot_file_url+'/B_result.csv', delimiter=',')
        
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='Training Accuracy')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='Validation Accuracy')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='green', label='Test Accesturacy')

    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    title ="Training Performance of Classifier: "+ name + plot_common_title + '\n'+"Test Accuracy: "+ str(test_accuracy)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_save_path+"User: "+ str(selected_user)+'Classifier: '+ name +" "+ title + '.png')
    
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='green', label='Training Loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (%)', fontsize=14)
    title = "Training Loss of Classifier: "+ name + plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_save_path+"User: "+ str(selected_user)+'Classifier: '+ name +" "+ title + '.png')
    
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='green', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (%)', fontsize=14)
    title = "Validation Loss of Classifier: "+ name + plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_save_path+"User: "+ str(selected_user)+'Classifier: '+ name +" "+ title + '.png')
    
    
def plot_activity(df, position='Default', save = False, save_url = ""):
    data = df[['AccX', 'AccY','AccZ']][:df.shape[0]]

    axis = data.plot(subplots=True, figsize=(16, 12), title=position)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
        if save == True:
            ax.figure.savefig(save_url+'.png') 
            
    
def plot_xyz(dataframe):
        
    plt.figure(figsize=(13,10))
    plt.plot(range(1, dataframe.shape[0]), dataframe[:, 0], color='blue', label='AccX')
#     plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='AccY')
#     plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='green', label='AccZ')
    

def A_plot():
    data = np.loadtxt('A_result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('A Training and Test Accuracy', fontsize=20)
    
def B_plot():
    data = np.loadtxt('B_result.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('B Training and Test Accuracy', fontsize=20)
            
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            
    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    
# def two_component_pca_visualization(data, ground_truth, target_no = 4):
#     pca_array = np.reshape(data, (data.shape[0],(data.shape[1]*data.shape[2]*data.shape[3])))
#     pca_array.shape
    
    
#     pca = PCA(n_components=2)
#     principalComponents = pca.fit_transform(pca_array)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['PC 1', 'PC 2'])
    
    
#     pca_gt = np.array(ground_truth)
#     pca_gt_Df = pd.DataFrame(data = pca_gt, columns = ['gt'])
    
#     final_pca_df = pd.concat([principalDf, pca_gt_Df], axis = 1)
    
#     fig = plt.figure(figsize = (10,10))
#     ax = fig.add_subplot(1,1,1) 
#     ax.set_xlabel('PC 1', fontsize = 15)
#     ax.set_ylabel('PC 2', fontsize = 15)
#     ax.set_title('2 component PCA', fontsize = 20)

#     global targets

#     if target_no == 7:
#         targets = [0,1,2,3,4,5,6]
#         target_label = ['Sitting','Standing','Lying','Walking','Running', 'Stair Up', 'Stair Down']
#         colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
#     elif target_no == 4:
#         targets = [0,1,2,3]
#         target_label = ['Sitting','Standing','Lying','Walking']
#         colors = ['r', 'g', 'b', 'y']
#     elif target_no == 3:
#         targets = [0,1,2]
#         target_label = ['Dominant Arm','Torso','Dominant Leg']
#         colors = ['r', 'g', 'b']
#     elif target_no == 11:
#         targets = [1,2,3,4,5,6,7,8,9,10,11]
#         target_label = ['Sitting','Standing','Lying','Walking','Running', 'cycling','Nordic walking','Stair Up', 'Stair Down', 'vacuum cleaning','ironing']
      
        
#     for target, color in zip(targets,colors):
#         indicesToKeep = final_pca_df['gt'] == target
#         ax.scatter(final_pca_df.loc[indicesToKeep, 'PC 1']
#                    , final_pca_df.loc[indicesToKeep, 'PC 2']
#                    #, c = color
#                    , s = 50)
#     ax.legend(target_label)
#     ax.grid()
    
    
# def three_component_pca_visualization(data, ground_truth, target_no = 4):
#     pca_array = np.reshape(data, (data.shape[0],(data.shape[1]*data.shape[2]*data.shape[3])))
#     pca_array.shape
    
#     pca3 = PCA(n_components=3)
#     principalComponents3 = pca3.fit_transform(pca_array)
#     principalDf3 = pd.DataFrame(data = principalComponents3, columns = ['PC 1', 'PC 2', 'PC 3'])
    
#     pca_gt = np.array(ground_truth)
#     pca_gt_Df = pd.DataFrame(data = pca_gt, columns = ['gt'])

#     final_pca_df3 = pd.concat([principalDf3, pca_gt_Df], axis = 1)
    
#     fig = plt.figure(figsize = (10,10))
#     ax = fig.add_subplot(111, projection='3d') 
#     ax.set_xlabel('PC 1', fontsize = 15)
#     ax.set_ylabel('PC 2', fontsize = 15)
#     ax.set_zlabel('PC 3', fontsize = 15)
#     ax.set_title('3 component PCA', fontsize = 20)
    
#     if target_no == 7:
#         targets = [0,1,2,3,4,5,6]
#         target_label = ['Sitting','Standing','Lying','Walking','Running', 'Stair Up', 'Stair Down']
#         colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
#     else:
#         targets = [0,1,2,3]
#         target_label = ['Sitting','Standing','Lying','Walking']
#         colors = ['r', 'g', 'b', 'y']

#     for target, color in zip(targets,colors):
#         indicesToKeep = final_pca_df3['gt'] == target
#         ax.scatter(final_pca_df3.loc[indicesToKeep, 'PC 1']
#                    , final_pca_df3.loc[indicesToKeep, 'PC 2']
#                    , final_pca_df3.loc[indicesToKeep, 'PC 3']
#                    , c = color
#                    , s = 50)
#     ax.legend(target_label)
#     ax.grid()
    
    
# def two_component_pca_visualization_11_class(source1, source1_gt, source2, source2_gt, target_no = 4):
def two_component_pca_visualization(source1, source1_gt, target_no = 4):
    pca = PCA(n_components=2)
    
    source1_array = np.reshape(source1, (source1.shape[0],(source1.shape[1]*source1.shape[2]*source1.shape[3])))
    s1_pc = pca.fit_transform(source1_array)
    s1_df = pd.DataFrame(data = s1_pc, columns = ['PC 1', 'PC 2'])
    s1_gt = np.array(source1_gt)
    s1_gt_Df = pd.DataFrame(data = s1_gt, columns = ['gt'])
    s1 = pd.concat([s1_df, s1_gt_Df], axis = 1)
    
    
#     source2_array = np.reshape(source2, (source2.shape[0],(source2.shape[1]*source2.shape[2]*source2.shape[3])))
#     s2_pc = pca.fit_transform(source2_array)
#     s2_df = pd.DataFrame(data = s2_pc, columns = ['PC 1', 'PC 2'])
#     s2_gt = np.array(source2_gt)
#     s2_gt_Df = pd.DataFrame(data = s2_gt, columns = ['gt'])
#     s2 = pd.concat([s2_df, s2_gt_Df], axis = 1)
    
    
    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    
    global targets

    if target_no == 7:
        targets = [0,1,2,3,4,5,6]
        target_label = ['Sitting','Standing','Lying','Walking','Running', 'Stair Up', 'Stair Down']
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
    elif target_no == 4:
        targets = [0,1,2,3]
        target_label = ['Sitting','Standing','Lying','Walking']
        colors = ['r', 'g', 'b', 'y']
    elif target_no == 11:
        colors = ['black', 'indianred', 'peru', 'yellowgreen', 'darkgreen', 'teal', 'steelblue','navy','indigo','violet','crimson']
        targets = [0,1,2,3,4,5,6,7,8,9,10]
        target_label = ['ironing', 'lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'NordicWalking','ascending', 'descending', 'vacuum']
          
        
    for target, color in zip(targets,colors):
        s1_indicesToKeep = s1['gt'] == target
#         s2_indicesToKeep = s2['gt'] == target
        ax.scatter(s1.loc[s1_indicesToKeep, 'PC 1'], s1.loc[s1_indicesToKeep, 'PC 2'], 
                   marker='x', s = 30, c = color)
#         ax.scatter(s2.loc[s2_indicesToKeep, 'PC 1'], s2.loc[s2_indicesToKeep, 'PC 2'], marker='s', s = 30, label='second')
        
    ax.legend(target_label)
    ax.grid()
    
def plot_data_distribution(s1, s2, target, activity_list):

    n_groups = len(activity_list)

    plt.figure(figsize=(12,8))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, s1, bar_width, alpha=opacity, color='b', label='Source1')
    rects2 = plt.bar(index + bar_width, s2, bar_width, alpha=opacity, color='g', label='Source2')
    rects3 = plt.bar(index + 2*bar_width, target, bar_width, alpha=opacity, color='m',label='Target')

    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Window Number', fontsize=14)
    plt.title('Window Distribution', fontsize=10)
    plt.xticks(index + bar_width, activity_list, rotation = 45)
    plt.legend()
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()
    
    
    
def plot_data_distribution_six_activities(s1, s2, target, activity_list):

    n_groups = len(activity_list)

    plt.figure(figsize=(12,8))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, s1, bar_width, alpha=opacity, color='b', label='Source1')
    rects2 = plt.bar(index + bar_width, s2, bar_width, alpha=opacity, color='g', label='Source2')
    rects3 = plt.bar(index + 2*bar_width, target, bar_width, alpha=opacity, color='m',label='Target')

    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Window Number', fontsize=14)
    plt.title('Window Distribution', fontsize=10)
    plt.xticks(index + bar_width, activity_list, rotation = 45)
    plt.legend()
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show() 
    
    
def plot_data_distribution_six_activities_single_source(s1, target, activity_list, save_dir):

    n_groups = len(activity_list)

    plt.figure(figsize=(12,8))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, s1, bar_width, alpha=opacity, color='b', label='Source1')
    rects3 = plt.bar(index + 2*bar_width, target, bar_width, alpha=opacity, color='m',label='Target')

    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Window Number', fontsize=14)
    plt.title('Window Distribution', fontsize=10)
    plt.xticks(index + bar_width, activity_list, rotation = 45)
    plt.legend()
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir+"Activity Distribution.png")
    
    plt.show()
    
    
def plot_data_distribution_six_activities_multi_source(s1, s2, target, activity_list, save_dir):

    n_groups = len(activity_list)

    plt.figure(figsize=(12,8))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, s1, bar_width, alpha=opacity, color='b', label='Source1')
    rects2 = plt.bar(index + bar_width, s2, bar_width, alpha=opacity, color='g', label='Source2')
    rects3 = plt.bar(index + 2*bar_width, target, bar_width, alpha=opacity, color='m',label='Target')

    plt.xlabel('Activity', fontsize=14)
    plt.ylabel('Window Number', fontsize=14)
    plt.title('Window Distribution', fontsize=10)
    plt.xticks(index + bar_width, activity_list, rotation = 45)
    plt.legend()
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir+"Activity Distribution.png")
    
    plt.show()
   