#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import preprocessing
import pandas as pd


def print_model_parameters(classifier_A, classifier_B = 'none'):
    for name, param in classifier_A.named_parameters():
        logging.warning("Name: {}".format(name))
        logging.warning("Param: {}".format(param.dtype))

    if not (classifier_B is None):
        for name in classifier_B.named_parameters():
            logging.warning("Name: {}".format(name))
            logging.warning("Param: {}".format(param.dtype))
            
def similarity_between_model_layers(model_A, model_B):        
    model_A_param_list= []
    model_A_param_layers_no = len(list(model_A.parameters()))
    for x in range(model_A_param_layers_no):
        a = list(model_A.parameters())[x].clone()
        model_A_param_list.append(a)

    model_B_param_list= []
    model_B_param_layers_no = len(list(model_B.parameters()))
    for x in range(model_B_param_layers_no):
        a = list(model_B.parameters())[x].clone()
        model_B_param_list.append(a) 

    for x in range(model_B_param_layers_no):
        logging.warning("Classifier A and Classifier B have Same Parameter in Layer {}:{}".format(x,torch.equal(model_A_param_list[x].data,model_B_param_list[x].data)))
        
def update_checking_between_model():
    # Needs to modify the function - "similarity_between_model_layers"
    for x in range(10):
        b = list(extractor.parameters())[x].clone()
        logging.warning("Adversarial Iteration: {}, At Extractor Layer: {}, Same Parameter:{}".format(gan_epoch,x,torch.equal(param_list[x].data, b.data)))

    for x in range(A_param_layers_no):
        b = list(discriminator_A.parameters())[x].clone()
        logging.warning("Adversarial Iteration: {}, At Discriminator-A Layer: {}, Same Parameter:{}".format(gan_epoch,x,torch.equal(discrim_A_param_list[x].data, b.data)))

    for x in range(B_param_layers_no):
        b = list(discriminator_B.parameters())[x].clone()
        logging.warning("Adversarial Iteration: {}, At Discriminator-B Layer: {}, Same Parameter:{}".format(gan_epoch,x,torch.equal(discrim_B_param_list[x].data, b.data)))

        
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1)).cuda(gpu_id)
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    
    
    '''
    data = Variable(torch.zeros(size, 1)).cuda(gpu_id)
    return data

def check_a_sample(sample, sample_target):
    predicted_label = F.log_softmax(sample[0], dim = 0)
    logging.warning("Item list: %s",sample[0])
    logging.warning("Predicted: %s", predicted_label)

    logging.warning("Sum: %.12f", sum(predicted_label))
    logging.warning("Max: %.12f", max(predicted_label))

    logging.warning("Target: %s", sample_target[0])


def plot_grad_flow(named_parameters, step, gan_epoch, batch):
    #             if (i+1) % plot_interval == 0:
#                 plot_grad_flow(discriminator_A.named_parameters(), step, gan_epoch, batch_size)

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
    plt.ylim(bottom = -0.001, top=0.08) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    string = "At Step "+str(step)+", Gan epoch: "+str(gan_epoch)+ ", batch: "+str(batch)
    plt.title("Gradient flow"+string)
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    
def dataset_labelling():
    print("Sitting = 0")
    print("Standing = 1")
    print("Lying = 2")
    print("Walking = 3")
    print("Running = 4")
    print("Ascending up = 5")
    print("Descending Down = 6")
    
def plot_activity(activity, df, position, i=1000):
    data = df[df['Activity'] == activity][['AccX', 'AccY','AccZ', 'GyrX', 'GyrY', 'GyrZ', 'MagX', 'MagY','MagZ']][:i]
    title_name = ""
    if position == 0:
        title_name = "Torso"
    elif position == 1:
        title_name = "Right Arm"
    elif position == 2:
        title_name = "Left Arm"
    elif position == 3:
        title_name = "Right Leg"
    elif position == 4:
        title_name = "Left Leg"
        
    axis = data.plot(subplots=True, figsize=(16, 12), title=title_name)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
        
def plot_datasets(df, activity_id, pos, i=500):
    plot_activity(activity_id, df, pos, i)


