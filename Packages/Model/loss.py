import torch
# import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
# def get_dis_loss(dis_fake, dis_real):
#     D_loss = - torch.mean(dis_fake ** 2) - torch.mean((dis_real - 1) ** 2)
#     return D_loss


# def get_confusion_loss(dis_common):
#     confusion_loss = torch.mean((dis_common - 0.5) ** 2)
#     return confusion_loss

EPSILON = 1e-8
def get_dis_loss(dis_source, dis_target):
    D_loss = - torch.mean(torch.log(dis_source + EPSILON)) - torch.mean(torch.log(1-dis_target + EPSILON))
    return D_loss

def get_confusion_loss(dis_common):
    confusion_loss = torch.mean(torch.log(dis_common + EPSILON)) + torch.mean(torch.log(1-dis_common + EPSILON))
    return confusion_loss

def get_cls_loss(pred, gt): 
    criterion = nn.CrossEntropyLoss()
    cls_loss = criterion(pred, gt)
    return cls_loss

def BCELossCalculation(source, source_no):
    length = len(source)
    t_length= len(target)

    logging.warning("S1 length: %d, Target length: %d",length, t_length)
    
    if source_no == 1:
        s1_error_fake = loss(s1_source, ones_target(length))
        s1_error_real = loss(s1_target, zeros_target(t_length))
        s1_t_dis_loss = s1_error_fake + s1_error_real
        logging.warning("S1 Disc loss: %s", s1_t_dis_loss.data)
        return s1_t_dis_loss
    else:
        s2_error_fake = loss(s2_source, ones_target(length))
        s2_error_real = loss(s2_target, zeros_target(t_length))
        s2_t_dis_loss = s2_error_fake + s2_error_real
        logging.warning("S2 Disc Loss: %s", s2_t_dis_loss.data)
        return s2_t_dis_loss
