import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from algo.pretrained.robot_transformer_ar import RobotTransformerAR
from algo.models.models import MLP

class RTActorCritic(nn.Module):

    def __init__(self, config, network_config, device, kwargs):

        nn.Module.__init__(self)
        self.network_config = network_config
        self.device = device
        actions_num =self.network_config.action_dim
        input_shape = kwargs.pop('value_input_shape')

        self.pc_to_value = config.train.ppo.point_cloud_input_to_value
        if config.get('pc_input', False) and self.pc_to_value:
            self.pc_begin, self.pc_end = kwargs.pop('point_cloud_index')

        self.value_grads_to_pointnet = config.train.ppo.value_grads_to_pointnet
        self.pc_num = self.network_config.pc_num
        self.scale_proprio = self.network_config.scale_proprio
        self.scale_action = self.network_config.scale_action


        mlp_input_shape = input_shape


        self.limits = {'upper': torch.tensor([6.2832, 2.0944, 6.2832, 3.9270, 6.2832, 3.1416, 6.2832, 0.4700, 1.6100, 1.7090, 1.6180, 1.3960,
                                        1.1630, 1.6440, 1.7190, 0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180],
                                                requires_grad=False, dtype=torch.float32, device=self.device),
                       'lower': torch.tensor([-6.2832, -2.0590, -6.2832, -0.1920, -6.2832, -1.6930, -6.2832, -0.4700, -0.1960, -0.1740, -0.2270,
                                   0.2630, -0.1050, -0.1890, -0.1620, -0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270]
                                           ,requires_grad=False, dtype=torch.float32, device=self.device)}


        self.actor = RobotTransformerAR(
            cfg= config)
 

        self.value_fn = nn.Sequential(
            nn.Linear(mlp_input_shape,512),
            nn.ELU(inplace=True),
            nn.Linear(512,256),
            nn.ELU(inplace=True),
            nn.Linear(256,128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1)
        ) #check this 

        self.logstd = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32))
        #backbone sharing between value and critic? can this be implemented here in some way? 
        #not doing for now
        nn.init.constant_(self.logstd[:7], torch.log(torch.tensor(kwargs['init_eps_arm'])))
        nn.init.constant_(self.logstd[7:], torch.log(torch.tensor(kwargs['init_eps_hand'])))

    def scale_q(self, q):
        """
        Scale the proprioceptive data to be between -1 and 1.
        """
        q = (q - self.limits['lower'].view((1,-1))) / (self.limits['upper'] - self.limits['lower'])
        q = 2 * q - 1
        return q

    @torch.no_grad()
    def get_action(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, value = self._actor_critic(obs_dict)
        sigma = torch.exp(self.logstd)
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
        mu, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        
        #what to do with the value network? 
        obs = obs_dict['obs']

        proprio_hist = obs_dict['proprio_buf']

        if self.scale_proprio:
            proprio_hist = self.scale_q(proprio_hist) #scale proprio hist 

        pc_hist = obs_dict['pc_buf'] #this is normalized


        attention_mask = obs_dict['attn_mask']
        timesteps = obs_dict['timesteps']

        if self.actor.cfg:
            action_hist = obs_dict['action_buf']
            action_hist = torch.cat((action_hist, torch.zeros_like(action_hist[:,:1,:])), dim=1)
        else:
            action_hist=None

        res_dict, pc_embed = self.actor(proprio_hist, pc_hist, action_hist, timesteps.long(), attention_mask)

        # Value function should reuse features?

        if not self.value_grads_to_pointnet:
            pc_embed = pc_embed.detach()

        if self.pc_to_value:
            obs = torch.cat([obs[:,:self.pc_begin],pc_embed[:,-1],obs[:,self.pc_end:]],dim=1)
        value = self.value_fn(obs)
       
        mu = res_dict['action'][:,-1]   #sigma in previous policy was independent of observations..F
        
        if not self.scale_action:
            mu = self.scale_q(mu)
            
        return mu, value

    def forward(self, input_dict):

        prev_actions = input_dict.get('prev_actions', None)
        mu, value = self._actor_critic(input_dict)
        sigma = torch.exp(self.logstd)
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
