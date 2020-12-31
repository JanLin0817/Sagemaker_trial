import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from easy_net import Net

def model_fn(model_dir):
    model = Net()
    model_path = os.path.join(model_dir, 'cifar_net_epoch_2.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model
