
import math

import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        if m.conv.weight is not None:
            if m.residual_init:
                init.xavier_uniform_(m.conv.weight.data, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(m.conv.weight.data)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias.data, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def print_network(model, name, out_file=None):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    if out_file is None:
        print(name)
        print(model)
        print('The number of parameters: {}'.format(num_params))
    else:
        with open(out_file, 'w') as f:
            f.write('{}\n'.format(name))
            f.write('{}\n'.format(model))
            f.write('The number of parameters: {}\n'.format(num_params))