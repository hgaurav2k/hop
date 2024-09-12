# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class SavingModel(nn.Module):
    "Saves the two models (runnig_mean_std and actor_critic) required for infence and simplifies TT code"
    def __init__(self, actor_critic_model, running_std_model):
        super(SavingModel, self).__init__()
        self.actor_critic_model = copy.deepcopy(actor_critic_model)
        self.running_std_model = copy.deepcopy(running_std_model)
        self.running_std_model.eval()

    def forward(self, x):
        x = self.running_std_model(x)
        input_dict = {'obs': x}
        mu = self.actor_critic_model.infer_action(input_dict)
        return mu

class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    def __init__(self):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        mlp_input_shape = input_shape

        out_size = self.units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def get_action(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def infer_action(self, obs_dict):
        # used during inference
        mu, _, _= self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        mu,logstd,value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        prev_actions = input_dict.get('prev_actions', mu.clone())
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma
        }

        return result



class PointNetActorCritic(nn.Module):

    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.pc_out_dim = kwargs.pop('point_cloud_out_dim')
        self.pc_begin, self.pc_end = kwargs.pop('point_cloud_index')
        self.pc_num = kwargs.pop('point_cloud_num')

        mlp_input_shape = input_shape
        out_size = self.units[-1]

        self.point_net = nn.Sequential(
            nn.Linear(3,self.pc_out_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.pc_out_dim,self.pc_out_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.pc_out_dim,self.pc_out_dim),
            nn.MaxPool2d((self.pc_num,1))
        )

        self.actor_mlp = MLP(units=self.units, input_size=self.pc_begin + self.pc_out_dim)
        self.obs_end_actor = self.pc_begin + self.pc_out_dim
        self.value = MLP(units=self.units, input_size=mlp_input_shape)
        self.value_final = nn.Linear(out_size, 1)
        # self.value = nn.Linear(out_size, 1)
        self.mu = nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def get_action(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def infer_action(self, obs_dict):
        # used during inference
        mu, _, _= self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):

        obs = obs_dict['obs']
        pc_info = obs[:,self.pc_begin:self.pc_end].reshape(-1,self.pc_num,3)
        pc_rep = self.point_net(pc_info).squeeze(1)
        obs = torch.cat([obs[:,:self.pc_begin],pc_rep,obs[:,self.pc_end:]],dim=1)
        x = self.actor_mlp(obs[:,:self.obs_end_actor])
        value_h = self.value(obs)
        value = self.value_final(value_h)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu,logstd,value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma
        }
        return result
