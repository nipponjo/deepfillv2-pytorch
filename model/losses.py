import torch

def ls_loss_d(pos, neg, value=1.):
    """
    gan with least-square loss
    """
    l2_pos = torch.mean((pos-value)**2)
    l2_neg = torch.mean(neg**2)
    d_loss = 0.5*l2_pos + 0.5*l2_neg 
    return d_loss

def ls_loss_g(neg, value=1.):    
    """
    gan with least-square loss
    """
    g_loss = torch.mean((neg-value)**2)
    return g_loss

def hinge_loss_d(pos, neg):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    hinge_pos = torch.mean(torch.relu(1-pos))
    hinge_neg = torch.mean(torch.relu(1+neg))
    d_loss = 0.5*hinge_pos + 0.5*hinge_neg   
    return d_loss

def hinge_loss_g(neg):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    g_loss = -torch.mean(neg)
    return g_loss