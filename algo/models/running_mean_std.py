# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,1,2]
            if len(self.insize) == 2:
                self.axis = [0,1] #make this 0 and 1? 
            if len(self.insize) == 1:
                self.axis = [0]
            self.in_size = self.insize[-1] 
        else:
            self.axis = [0]
            self.in_size = insize

        self.register_buffer('running_mean', torch.zeros(self.in_size, dtype = torch.float64))
        self.register_buffer('running_var', torch.ones(self.in_size, dtype = torch.float64))
        self.register_buffer('count', torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )
            
        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, 1, 1, self.in_size]).expand_as(input)
                current_var = self.running_var.view([1, 1, 1, self.in_size]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, 1, self.in_size]).expand_as(input)
                current_var = self.running_var.view([1, 1, self.in_size]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.in_size]).expand_as(input)
                current_var = self.running_var.view([1, self.in_size]).expand_as(input)        
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output
                                                                                                                                                                                                                                                                                                                                 

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y
