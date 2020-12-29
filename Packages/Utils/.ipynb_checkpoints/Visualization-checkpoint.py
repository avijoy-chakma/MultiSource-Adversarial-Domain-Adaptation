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


import numpy as np
import matplotlib.pyplot as plt


def plot_agreed_above_threshold_incorrect_list(pseudo_save_path, plot_common_title):
    data = np.loadtxt(pseudo_save_path+'agreed_above_threshold_incorrect_list.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Incorrect S1 A1')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='Incorrect S2 A1')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='gray', label='Incorrect S1 A2')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='goldenrod', label='Incorrect S2 A2')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='red', label='Incorrect S1 A3')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='green', label='Incorrect S2 A3')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='black', label='Incorrect S1 A4')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='orange', label='Incorrect S2 A4')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold (%)', fontsize=14)
    title = "Agreed Above Threshold Incorrect Sample Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Agreed Above Threshold Incorrect Sample Analysis.png')
    
    
    
def plot_agreed_different_threshold_s1_correct_incorrect_list(pseudo_save_path, plot_common_title):
    data = np.loadtxt(pseudo_save_path+'agreed_different_threshold_s1_correct_incorrect_list.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Correct S1 A1')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='Incorrect S1 A1')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='gray', label='Correct S1 A2')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='goldenrod', label='Incorrect S1 A2')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='red', label='Correct S1 A3')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='green', label='Incorrect S1 A3')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='black', label='Correct S1 A4')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='orange', label='Incorrect S1 A4')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold (%)', fontsize=14)
    title = "Agreed different threshold s1 correct incorrect analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Agreed different threshold s1 correct incorrect analysis.png')
    
    
    
def plot_agreed_different_threshold_s2_correct_incorrect_list(pseudo_save_path, plot_common_title):
    data = np.loadtxt(pseudo_save_path+'agreed_different_threshold_s2_correct_incorrect_list.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Correct S2 A1')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='Incorrect S2 A1')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='gray', label='Correct S2 A2')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='goldenrod', label='Incorrect S2 A2')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='red', label='Correct S2 A3')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='green', label='Incorrect S2 A3')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='black', label='Correct S2 A4')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='orange', label='Incorrect S2 A4')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold (%)', fontsize=14)
    title = "Agreed different threshold s2 correct incorrect analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Agreed different threshold s2 correct incorrect analysis.png')
    
    
    
def plot_disagreed_s1_correct_incorrect_list(pseudo_save_path, plot_common_title):
    data = np.loadtxt(pseudo_save_path+'disagreed_s1_correct_incorrect_list.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Disagreed Correct S1 A1')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='Disagreed Incorrect S1 A1')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='gray', label=' Disagreed Correct S1 A2')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='goldenrod', label='Disagreed Incorrect S1 A2')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='red', label='Disagreed Correct S1 A3')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='green', label='Disagreed Incorrect S1 A3')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='black', label='Disagreed Correct S1 A4')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='orange', label='Disagreed Incorrect S1 A4')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold (%)', fontsize=14)
    title = "Disagreed s1 correct incorrect analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Disagreed s1 correct incorrect analysis.png')
    
def plot_disagreed_s2_correct_incorrect_list(pseudo_save_path, plot_common_title):
    data = np.loadtxt(pseudo_save_path+'disagreed_s2_correct_incorrect_list.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Disagreed Correct S2 A1')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='Disagreed Incorrect S2 A1')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='gray', label=' Disagreed Correct S2 A2')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='goldenrod', label='Disagreed Incorrect S2 A2')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='red', label='Disagreed Correct S2 A3')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='green', label='Disagreed Incorrect S2 A3')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='black', label='Disagreed Correct S2 A4')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='orange', label='Disagreed Incorrect S2 A4')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold (%)', fontsize=14)
    title = "Disagreed s1 correct incorrect analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Disagreed s2 correct incorrect analysis.png')
    

def plot_selection_distribution(save_path, plot_common_title, steps):
    data = np.loadtxt(save_path+'selection_distribution.csv', delimiter=',')
    sources = ['S1', 'S2']
    
    plt.figure(figsize=(13,10))
    
    steps = [i+1 for i in range(steps)]
    
    data = [
        data[:, 0].tolist(),
        data[:, 1].tolist()
    ]
    
    
    plot_stacked_bar(
        data, 
        sources, 
        plot_common_title,
        category_labels=steps, 
        show_values=False, 
        value_format="{:.2f}",
        colors=['tab:blue', 'tab:green'],
        y_label="Source Selection Distribution"
    )
    
    plt.savefig(save_path + 'Source Selection Distribution.png')

    
    

def plot_disc_deconv_loss_track(save_path, plot_common_title):
    data = np.loadtxt(save_path+'losses.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='S1 Classification')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='S2 Classification')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='dodgerblue', label='S1 Deconv')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='gray', label='S2 Deconv')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='goldenrod', label='Target Deconv - S1')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='black', label='Target Deconv - S2')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (%)', fontsize=14)
    title = "Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path + 'Classification and Deconv Loss Analysis.png')
    
    
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='maroon', label='S1 Target Conf Loss')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='seagreen', label='S2 Target Conf Loss')
    plt.plot(range(1, len(data[:, 8]) + 1), data[:, 8], color='dodgerblue', label='S1 Target Disc Loss')
    plt.plot(range(1, len(data[:, 9]) + 1), data[:, 9], color='gray', label='S2 Target Disc Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (%)', fontsize=14)
    title = "Discriminator and Confusion Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path + 'Discriminator and Confusion Loss Analysis.png')
    
    
def plot_disagreed_pseudo_distribution_np_track(pseudo_save_path, plot_common_title, steps, activities):
    data = np.loadtxt(pseudo_save_path+'/disagreed_pseudo_distribution_np_track.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Total Disagreed Samples')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='Disagreed, S1 Correct Prediction')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='dodgerblue', label='Disagreed, S1 Incorrect Prediction')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Disagreed Pseudo Sample Number (%)', fontsize=14)
    title = "Disagreed-S1 Pseudo Label Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Disagreed-S1 Pseudo Label Analysis.png')
    
    
    # Source 2
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='Total Disagreed Samples')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='seagreen', label='Disagreed, S2 Correct Prediction')
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='dodgerblue', label='Disagreed, S2 Incorrect Prediction')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Disagreed Pseudo Sample Number (%)', fontsize=14)
    title = "Disagreed-S2 Pseudo Label Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Disagreed-S2 Pseudo Label Analysis.png')
    
    

def plot_disagreed_pseudo_sample_threshold_analysis_np_track(pseudo_save_path, plot_common_title, steps, activities):
    data = np.loadtxt(pseudo_save_path+'/disagreed_pseudo_sample_analysis_np_track.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='maroon', label='Disagreed S1 A1')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='seagreen', label='Disagreed S1 A2')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='gray', label='Disagreed S1 A3')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='goldenrod', label='Disagreed S1 A4')
    
    plt.plot(range(1, len(data[:, 12]) + 1), data[:, 12], color='red', label='Disagreed S2 A1')
    plt.plot(range(1, len(data[:, 13]) + 1), data[:, 13], color='green', label='Disagreed S2 A2')
    plt.plot(range(1, len(data[:, 14]) + 1), data[:, 14], color='black', label='Disagreed S2 A3')
    plt.plot(range(1, len(data[:, 15]) + 1), data[:, 15], color='orange', label='Disagreed S2 A4')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold Value (%)', fontsize=14)
    title = "Disagreement Classwise-Threshold Value Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Disagreement Classwise-Threshold Value Analysis.png')
    
    
def plot_disagreed_pseudo_sample_analysis_np_track(pseudo_save_path, plot_common_title, steps, activities):
    pseudo_disagreed_sample_distribution = np.loadtxt(pseudo_save_path+'/disagreed_pseudo_sample_analysis_np_track.csv', delimiter=',')
    
    steps = [i+1 for i in range(steps)]
    
    plt.figure(figsize=(13,10))
    
    if len(activities) == 4:
        s1data = [
            pseudo_disagreed_sample_distribution[:, 0].tolist(),
            pseudo_disagreed_sample_distribution[:, 1].tolist(),
            pseudo_disagreed_sample_distribution[:, 2].tolist(),
            pseudo_disagreed_sample_distribution[:, 3].tolist()
        ]
    
    
    plot_stacked_bar(
        s1data, 
        activities, 
        plot_common_title,
        category_labels=steps, 
        show_values=False, 
        value_format="{:.2f}",
        colors=['tab:orange', 'tab:purple', 'tab:blue', 'tab:green'],
        y_label="Disagreed S1 Correctly Predicted Pseudo Sample Number"
    )
    
    plt.savefig(pseudo_save_path + 'Disagreed S1 Correctly Predicted Pseudo Sample Number.png')
    
    
    # S2 Predicted Pseudo Analysis
    plt.figure(figsize=(13,10))
    
    if len(activities) == 4:
        s2data = [
            pseudo_disagreed_sample_distribution[:, 8].tolist(),
            pseudo_disagreed_sample_distribution[:, 9].tolist(),
            pseudo_disagreed_sample_distribution[:, 10].tolist(),
            pseudo_disagreed_sample_distribution[:, 11].tolist()
        ]
    
    
    plot_stacked_bar(
        s2data, 
        activities, 
        plot_common_title,
        category_labels=steps, 
        show_values=False, 
        value_format="{:.2f}",
        colors=['tab:orange', 'tab:purple', 'tab:blue', 'tab:green'],
        y_label="Disagreed S2 Correctly Predicted Pseudo Sample Number"
    )
    
    plt.savefig(pseudo_save_path + 'Disagreed S2 Correctly Predicted Pseudo Sample Number.png')
    
    

def plot_incorrect_pseudo_threshold_distribution(pseudo_save_path, plot_common_title, steps, activities):
    
    data = np.loadtxt(pseudo_save_path+'/incorrect_pseudo_threshold_distribution_track.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='maroon', label='S1 A1')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='seagreen', label='S1 A2')
    plt.plot(range(1, len(data[:, 2]) + 1), data[:, 2], color='gray', label='S1 A3')
    plt.plot(range(1, len(data[:, 3]) + 1), data[:, 3], color='goldenrod', label='S1 A4')
    
    plt.plot(range(1, len(data[:, 4]) + 1), data[:, 4], color='red', label='S2 A1')
    plt.plot(range(1, len(data[:, 5]) + 1), data[:, 5], color='green', label='S2 A2')
    plt.plot(range(1, len(data[:, 6]) + 1), data[:, 6], color='black', label='S2 A3')
    plt.plot(range(1, len(data[:, 7]) + 1), data[:, 7], color='orange', label='S2 A4')
    
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Threshold Value (%)', fontsize=14)
    title = "Classwise-Threshold Value Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(pseudo_save_path + 'Classwise-Threshold Value Analysis.png')

    
def plot_correct_pseudo_distribution(pseudo_save_path, plot_common_title, steps, activities):
    activities.append("Incorrect-Pseudo-Samples")
    pseudo_distribution_track_data = np.loadtxt(pseudo_save_path+'/correct_pseudo_distribution_track.csv', delimiter=',')
    
    steps = [i+1 for i in range(steps)]
    
    plt.figure(figsize=(13,10))
    
    if len(activities) == 5:
        data = [
            pseudo_distribution_track_data[:, 0].tolist(),
            pseudo_distribution_track_data[:, 1].tolist(),
            pseudo_distribution_track_data[:, 2].tolist(),
            pseudo_distribution_track_data[:, 3].tolist(),
            pseudo_distribution_track_data[:, 4].tolist()
        ]

    plot_stacked_bar(
        data, 
        activities, 
        plot_common_title,
        category_labels=steps, 
        show_values=False, 
        value_format="{:.2f}",
        colors=['tab:orange', 'tab:purple', 'tab:blue', 'tab:green', 'tab:red'],
        y_label="Correct Pseudo Sample Number"
    )

    plt.savefig(pseudo_save_path + 'Pseudo Label Distribution Analysis.png')
    activities.remove("Incorrect-Pseudo-Samples")
#     plt.show()

    
    
    
def plot_stacked_bar(data, series_labels, plot_common_title, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, 
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
#     plt.ylabel('Correct Pseudo Sample Number (%)', fontsize=14)
    title = "Pseudo Label Distribution Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")
                
                
def plot_pseudo_agreement_track(plot_file_url, plot_common_title):
    
    pseudo_track_data = np.loadtxt(plot_file_url+'/agreement_pseudo_track.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(pseudo_track_data[:, 0]) + 1), pseudo_track_data[:, 0], color='maroon', label='Total Target Samples')
    plt.plot(range(1, len(pseudo_track_data[:, 1]) + 1), pseudo_track_data[:, 1], color='seagreen', label='Total Predicted Samples')
    plt.plot(range(1, len(pseudo_track_data[:, 2]) + 1), pseudo_track_data[:, 2], color='dodgerblue', label='Total Correctly Predicted')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Correct Pseudo Sample Number (%)', fontsize=14)
    title = "Pseudo Label Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Pseudo Label Analysis.png')
    
    
def plot_class_disc_loss_no_augmentation(plot_file_url, plot_common_title):
    
    class_loss_data = np.loadtxt(plot_file_url+'/class_loss.csv', delimiter=',')
    disc_loss_data = np.loadtxt(plot_file_url+'/disc_loss.csv', delimiter=',')
    
    # Classification loss
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(class_loss_data[:, 0]) + 1), class_loss_data[:, 0], color='maroon', label='S1 Class Loss')
    plt.plot(range(1, len(class_loss_data[:, 1]) + 1), class_loss_data[:, 1], color='seagreen', label='S2 Class Loss')
    plt.plot(range(1, len(class_loss_data[:, 2]) + 1), class_loss_data[:, 2], color='black', label='S1 Pseudo Loss')
    plt.plot(range(1, len(class_loss_data[:, 3]) + 1), class_loss_data[:, 3], color='goldenrod', label='S2 Pseudo Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Classifier Loss Value (%)', fontsize=14)
    title = "Classifier Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Classification Loss.png')
    
    # Discrimiator Losses
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(disc_loss_data[:, 0]) + 1), disc_loss_data[:, 0], color='orange', label='S1 Disc Loss')
    plt.plot(range(1, len(disc_loss_data[:, 1]) + 1), disc_loss_data[:, 1], color='teal', label='S2 Disc Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Discriminator Loss Value (%)', fontsize=14)
    title = "Discriminator Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Discriminator Loss.png')
    
def plot_losses(plot_file_url, plot_common_title):
    
    a_data = np.loadtxt(plot_file_url+'/losses.csv', delimiter=',')
    
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(a_data[:, 0]) + 1), a_data[:, 0], color='maroon', label='S1 Class Loss')
    plt.plot(range(1, len(a_data[:, 1]) + 1), a_data[:, 1], color='seagreen', label='S2 Class Loss')
    plt.plot(range(1, len(a_data[:, 2]) + 1), a_data[:, 2], color='navy', label='S1 Disc Loss')
    plt.plot(range(1, len(a_data[:, 3]) + 1), a_data[:, 3], color='orange', label='S2 Disc Loss')
    plt.plot(range(1, len(a_data[:, 4]) + 1), a_data[:, 4], color='teal', label='S1 Conf Loss')
    plt.plot(range(1, len(a_data[:, 5]) + 1), a_data[:, 5], color='purple', label='S2 Conf Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss Value (%)', fontsize=14)
    title = "Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Loss.png')

def plot_pseudo_track(plot_file_url, plot_common_title):
    
    pseudo_track_data = np.loadtxt(plot_file_url+'/pseudo_track.csv', delimiter=',')
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(pseudo_track_data[:, 0]) + 1), pseudo_track_data[:, 0], color='maroon', label='S1 Pseudo and Correction')
    plt.plot(range(1, len(pseudo_track_data[:, 1]) + 1), pseudo_track_data[:, 1], color='seagreen', label='S2 Pseudo and Correction')
    plt.plot(range(1, len(pseudo_track_data[:, 2]) + 1), pseudo_track_data[:, 2], color='dodgerblue', label='S1 Pseudo and Ground')
    plt.plot(range(1, len(pseudo_track_data[:, 3]) + 1), pseudo_track_data[:, 3], color='teal', label='S2 Pseudo and Ground')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Correct Pseudo Sample Number (%)', fontsize=14)
    title = "Pseudo Label Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Pseudo Label Analysis.png')
    
    

def plot_class_disc_loss(plot_file_url, plot_common_title):
    
    class_loss_data = np.loadtxt(plot_file_url+'/class_loss.csv', delimiter=',')
    disc_loss_data = np.loadtxt(plot_file_url+'/disc_loss.csv', delimiter=',')
    
    # Classification loss
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(class_loss_data[:, 0]) + 1), class_loss_data[:, 0], color='maroon', label='S1 Class Loss')
    plt.plot(range(1, len(class_loss_data[:, 1]) + 1), class_loss_data[:, 1], color='seagreen', label='S2 Class Loss')
    plt.plot(range(1, len(class_loss_data[:, 2]) + 1), class_loss_data[:, 2], color='dodgerblue', label='S1 Aug Loss')
    plt.plot(range(1, len(class_loss_data[:, 3]) + 1), class_loss_data[:, 3], color='teal', label='S2 Aug Loss')
    plt.plot(range(1, len(class_loss_data[:, 4]) + 1), class_loss_data[:, 4], color='black', label='S1 Pseudo Loss')
    plt.plot(range(1, len(class_loss_data[:, 5]) + 1), class_loss_data[:, 5], color='goldenrod', label='S2 Pseudo Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Classifier Loss Value (%)', fontsize=14)
    title = "Classifier Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Classification Loss.png')
    
    # Discrimiator Losses
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(disc_loss_data[:, 0]) + 1), disc_loss_data[:, 0], color='orange', label='S1 Disc Loss')
    plt.plot(range(1, len(disc_loss_data[:, 1]) + 1), disc_loss_data[:, 1], color='teal', label='S2 Disc Loss')
    plt.plot(range(1, len(disc_loss_data[:, 2]) + 1), disc_loss_data[:, 2], color='purple', label='S1 Aug Disc Loss')
    plt.plot(range(1, len(disc_loss_data[:, 3]) + 1), disc_loss_data[:, 3], color='black', label='S2 Aug Disc Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Discriminator Loss Value (%)', fontsize=14)
    title = "Discriminator Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Discriminator Loss.png')
    

def plot_aug_seg_loss(plot_file_url, plot_common_title):
    
    a_data = np.loadtxt(plot_file_url+'/A_result.csv', delimiter=',')
    b_data = np.loadtxt(plot_file_url+'/B_result.csv', delimiter=',')
    
    
    plt.figure(figsize=(13,10))
    plt.plot(range(1, len(a_data[:, 2]) + 1), a_data[:, 2], color='maroon', label='A Training Loss')
    plt.plot(range(1, len(a_data[:, 4]) + 1), a_data[:, 4], color='seagreen', label='A Segmentation Loss')
    plt.plot(range(1, len(a_data[:, 5]) + 1), a_data[:, 5], color='navy', label='A Augmentation Loss')
    
    plt.plot(range(1, len(b_data[:, 2]) + 1), b_data[:, 2], color='orange', label='B Training Loss')
    plt.plot(range(1, len(b_data[:, 4]) + 1), b_data[:, 4], color='teal', label='B Segmentation Loss')
    plt.plot(range(1, len(b_data[:, 5]) + 1), b_data[:, 5], color='purple', label='B Augmentation Loss')
    
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss Value (%)', fontsize=14)
    title = "Classifier Loss Analysis: "+ plot_common_title
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file_url + 'Loss.png')
    
    
    
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
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 12})
    
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

    
# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           cmap=None,
#                           normalize=True):

#     plt.style.use('ggplot')
#     plt.rcParams.update({'font.size': 12})

#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     plt.figure(figsize=(6, 6))

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.colorbar(fraction=0.046, pad=0.04)

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('Ground Truth')
#     plt.xlabel('Prediction')
#     plt.show()
    
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
   