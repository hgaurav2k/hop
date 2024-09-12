# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

from mimetypes import common_types
import os
import time
import torch
from copy import deepcopy 
from algo.ppo_transformer.experience import ExperienceBuffer
from algo.ppo_transformer.mem_eff_experience import MemoryEfficientExperienceBuffer
from algo.models.models import  SavingModel
from algo.models.running_mean_std import RunningMeanStd
from utils.misc import AverageScalarMeter
import utils.pytorch_utils as ptu 
import torch.nn as nn 
from datetime import datetime
from termcolor import cprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import jit
import numpy as np
import pickle
import cv2 
from PIL import Image 
import PIL 
from algo.models.rt_actor_critic import RTActorCritic
from algo.models.pt_actor_critic import PTActorCritic
import json 
class PPOTransformer(nn.Module):

    def __init__(self, env, config, logger=None, rank=0):
        super().__init__()

        if config.num_gpus > 1:
            self.device = f"cuda:{rank}" #config['rl_device']
        else:
            self.device = config['rl_device']
        
        self.network_config = config.train.network
        self.ppo_config = config.train.ppo
        self.config = config
        self.logger = logger 

        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.num_gradient_steps = self.ppo_config['num_gradient_steps']
        self.warmup_critic_gradsteps = self.ppo_config['critic_warmup_steps'] #-1 if no critic warmup
        self.init_eps_arm = self.ppo_config.get('initEpsArm')
        self.init_eps_hand = self.ppo_config.get('initEpsHand')
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.rank = rank
        self.num_gpus = config.num_gpus

        self.log_video = self.env.log_video

        # if config['pc_input'] and config["enableProprioHistory"]:
        #     net_config = {
        #         'actor_units': self.network_config.mlp.units,
        #         'actions_num': self.actions_num,
        #         'point_cloud_index': (self.env.point_cloud_begin_index, self.env.point_cloud_end_index),
        #         'point_cloud_out_dim' : self.network_config.pc_mlp.out_dim,
        #         'point_cloud_num': self.env.point_cloud_sampled_dim,
        #         'input_shape': self.obs_shape[0] - self.env.point_cloud_sampled_dim*3 +  self.network_config.pc_mlp.out_dim,
        #     }
        #     self.model = PointNetActorCritic(net_config)
        
        #has to have an actor critic
        kwargs = {} 

        if config.train.ppo.point_cloud_input_to_value:
            kwargs['value_input_shape'] = self.network_config.hidden_dim + self.env.full_state_size - self.env.point_cloud_sampled_dim*3
            kwargs['point_cloud_index'] = (self.env.point_cloud_begin_index, self.env.point_cloud_end_index)
        else:
            kwargs['value_input_shape'] = self.env.full_state_size
        
        kwargs['point_cloud_input_to_value'] = config.train.ppo.point_cloud_input_to_value
        kwargs['init_eps_arm'] = self.init_eps_arm
        kwargs['init_eps_hand'] = self.init_eps_hand

        # if self.network_config.full_autoregressive:
        self.model = RTActorCritic(config, self.network_config, device = self.device, kwargs=kwargs)
        # else:
        #     self.model = PTActorCritic(self.network_config,kwargs)

        # if config.wandb_activate and rank == 0:
        #    wandb.watch(self.model,log="all",log_freq=1)

        # construct the local model

        self.model.to(self.device)

        # construct the DDP model
        if self.num_gpus > 1:
            self.ddp_model = DDP(self.model, device_ids=[rank],output_device=rank, find_unused_parameters=True)

        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_proprio_hist = self.ppo_config['normalize_proprio_hist']
        self.normalize_value = self.ppo_config['normalize_value']
        self.normalize_pc = self.ppo_config['normalize_pc']
        self.use_memory_efficient_buffer = self.ppo_config['useMemoryEfficientBuffer']
        # ---- Normalization ----
        # self.running_mean_std = RunningMeanStd(self.obs_shape[0]).to(self.device)
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd((self.obs_shape[0],)).to(self.device)  
        if self.normalize_proprio_hist:
            #for now size is same obs_shape
            self.proprio_mean_std = RunningMeanStd((self.network_config.context_length,
                                                    self.network_config.proprio_dim,),
                                                   per_channel=True).to(self.device)
        if self.normalize_pc:
            self.pc_mean_std = RunningMeanStd((self.network_config.context_length,self.env.point_cloud_sampled_dim,3)).to(self.device)

        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        if self.rank == 0 or self.num_gpus == 1:
            if logger is not None: 
                self.output_dir = os.path.join(logger._log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M'))
                self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
                #self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
                os.makedirs(self.nn_dir, exist_ok=True)
        #os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)

        if self.num_gpus > 1:
            self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), self.last_lr, weight_decay=self.weight_decay)
            #self.optimizer = torch.optim.AdamW(self.ddp_model.parameters(), self.last_lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or config.test or config.teacher_mode
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.min_lr = self.ppo_config['min_lr']
        self.max_lr = self.ppo_config['max_lr']
        self.scheduler = AdaptiveScheduler(self.kl_threshold, min_lr = self.min_lr, max_lr = self.max_lr)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Wandb Logger Info ----
        self.extra_info = {}

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        
        if self.use_memory_efficient_buffer:

            self.replay_buffer = MemoryEfficientExperienceBuffer(self.num_actors, 
                                            self.horizon_length, 
                                            self.batch_size, 
                                            self.minibatch_size, 
                                            self.num_gradient_steps,
                                            self.obs_shape[0], 
                                            self.network_config.proprio_dim,
                                            self.actions_num, 
                                            self.env.point_cloud_sampled_dim,
                                            self.network_config.context_length, 
                                            self.device)
        else:
            self.replay_buffer = ExperienceBuffer(self.num_actors, 
                                                self.horizon_length, 
                                                self.batch_size, 
                                                self.minibatch_size, 
                                                self.num_gradient_steps,
                                                self.obs_shape[0], 
                                                self.network_config.proprio_dim,
                                                self.actions_num, 
                                                self.env.point_cloud_sampled_dim,
                                                self.network_config.context_length, 
                                                self.device)

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.gradient_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        self.best_success = -10
        self.cur_success = -100
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0


        if config.get('teacher_mode',False):
            self.set_eval()
            self.model.actor.load_state_dict(torch.load(config.checkpoint,map_location=self.device))
            print("PPO Agent set to test mode")


    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_proprio_hist:
            self.proprio_mean_std.eval()
        if self.normalize_pc:
            self.pc_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        #add this later 
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_proprio_hist:
            self.proprio_mean_std.train()
        if self.normalize_pc:
            self.pc_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    @torch.no_grad()
    def get_action(self, obs_dict):
        
        if self.normalize_input:
            processed_obs = self.running_mean_std(obs_dict['obs'])
        else:
            processed_obs = obs_dict['obs']
        
        if self.normalize_proprio_hist:
            processed_proprio = self.proprio_mean_std(obs_dict['proprio_buf'].clone())
        else:
            processed_proprio = obs_dict['proprio_buf'].clone() #POLICY SPECIFIC SCALING: BEWARE OF THE ACTION SPACE

        if self.normalize_pc:
            processed_pc_hist = self.pc_mean_std(obs_dict['pc_buf'].clone())
        else:
            processed_pc_hist = obs_dict['pc_buf'].clone()

        input_dict = {
            "obs": processed_obs,
            "pc_buf": processed_pc_hist,
            "proprio_buf": processed_proprio,
            "action_buf": obs_dict['action_buf'],
            "attn_mask": obs_dict['attn_mask'],
            "timesteps": obs_dict['timesteps']
        }

        res_dict = self.model.get_action(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True) #why is this done?
        return res_dict

    def train(self):

        self.obs = self.env.reset()
        # self.obs = dict(obs=torch.zeros(self.obs['obs']))

        self.agent_steps = self.batch_size

        if self.log_video and self.rank == 0:
            steps_since_last_video = 0
            self.video_log_freq = 100e6
            print(f"Capturing video from simulation")
            self.env.start_video_recording()
            self.test()
            video_frames = self.env.stop_video_recording()
            fps = int(1/(self.env.control_freq_inv * self.env.sim_params.dt))
            self.logger.log_video(video_frames, name="Test Performance", fps=fps, step=(self.agent_steps*self.num_gpus))
            self.env.video_frames = []
            self.obs = self.env.reset()

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            train_info = self.train_epoch()
            self.replay_buffer.data_dict = None

            #log and save checkpoints
            if self.rank == 0 or self.num_gpus == 1:
                mean_rewards = self.episode_rewards.get_mean()
                mean_lengths = self.episode_lengths.get_mean()
                checkpoint_name = f'ep_{self.epoch_num}_step_{int( (self.agent_steps*self.num_gpus) // 1e6):04}M_reward_{mean_rewards:.2f}'
                if self.logger._summ_writer:
                    # TODO: Add number of GPU
                    self.logger.log_scalar(mean_rewards,'episode_rewards/step', self.agent_steps*self.num_gpus)
                    self.logger.log_scalar(mean_lengths,'episode_lengths/step', self.agent_steps*self.num_gpus)
                    self.logger.log_dict(self.extra_info, self.agent_steps*self.num_gpus)

                if self.save_freq > 0:
                    if (self.epoch_num-1) % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        self.save(os.path.join(self.nn_dir, 'last'))

                if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                    print(f'save current best reward: {mean_rewards:.2f}')
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, 'best'))

                if self.cur_success > self.best_success and self.epoch_num >= self.save_best_after:
                    print(f'save current best success: {self.cur_success:.2f}')
                    self.best_success = self.cur_success
                    self.save(os.path.join(self.nn_dir, 'best'))
                
                if self.log_video and self.rank == 0:
                    steps_since_last_video += self.batch_size * self.num_gpus
                    if steps_since_last_video > self.video_log_freq:
                        steps_since_last_video = 0
                        print(f"Capturing video from simulation")
                        self.env.start_video_recording()
                        self.test()
                        video_frames = self.env.stop_video_recording()
                        fps = int(1/(self.env.control_freq_inv * self.env.sim_params.dt))
                        self.logger.log_video(video_frames, name="Test Performance", fps=fps, step=(self.agent_steps*self.num_gpus))
                        self.env.video_frames = []
                        # After video, need to reset the buffers
                        self.obs = self.env.reset()
                        self.current_lengths = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
                        self.current_rewards = torch.zeros((self.num_actors, 1), dtype=torch.float32, device=self.device)

            

                print("Agent Steps: ", self.agent_steps, "Num Epoch: ", self.epoch_num, "Mean Reward: ", mean_rewards, "Mean Length: ", mean_lengths)

            #self.logger.flush()
            self.extra_info = {}

            # sync barrier (I assume this is done to wait for state syncronization)
            if self.num_gpus > 1:
                torch.distributed.barrier()

        print('***Max steps achieved***')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.normalize_input:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_proprio_hist:
            weights['proprio_mean_std'] = self.proprio_mean_std.state_dict()
        if self.normalize_pc:
            weights['pc_mean_std'] = self.pc_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')


    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn,map_location=self.device)
        # cprint('remember to remove stric!!!', 'red', attrs=['bold'])
        self.model.actor.load_state_dict(checkpoint) #, strict=False)
        cprint('model restored from {}'.format(fn), 'green', attrs=['bold'])
        if self.normalize_input:
            try:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            except:
                print("Could not restore running mean std for value function")
        if self.normalize_proprio_hist:
            self.proprio_mean_std.load_state_dict(checkpoint['proprio_mean_std'])
        if self.normalize_pc:
            self.pc_mean_std.load_state_dict(checkpoint['pc_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn,map_location=self.device)
        print(checkpoint.keys)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_proprio_hist:
            self.proprio_mean_std.load_state_dict(checkpoint['proprio_mean_std'])
        if self.normalize_pc:
            self.pc_mean_std.load_state_dict(checkpoint['pc_mean_std'])
        
        cprint('model restored from {}'.format(fn), 'green', attrs=['bold'])
        save_jit = self.config['save_jit']
        if save_jit:
            self.save_jit_trace(''.join(fn.split('.')[:-1]))
            self.save_joint_limits(''.join(fn.split('.')[:-1]))

    def save_jit_trace(self, name):
        print("Saving Jit Trace")
        self.running_mean_std.eval()
        saving_model = SavingModel(self.model, self.running_mean_std)
        obs = torch.ones(self.obs_shape, dtype=torch.float32).to(self.device)
        model_trace = jit.trace(saving_model, obs)
        jit.save(model_trace, f'{name}.jit.pt')
    
    def save_joint_limits(self, name):
        limits = {'lower': self.env.arm_dof_lower_limits.cpu().numpy(),
                  'upper': self.env.arm_dof_upper_limits.cpu().numpy()}
        with open(f'{name}.limits.pickle', 'wb') as handle:
            pickle.dump(limits, handle)
        return
        

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        if self.use_memory_efficient_buffer:
            self.play_steps_efficient_buffer()
        else:
            self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # print("Data collection time" , (time.time() - _t))
        # update network
        _t = time.time()
        self.set_train()
        a_losses, c_losses = [], []
        entropies, kls = [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []

            for i in range(len(self.replay_buffer)):

                
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, proprio_hist, pc_hist, action_hist, attn_mask, timesteps = self.replay_buffer[i] #i-th "mini-batch"

                #add this later
                if self.normalize_input: 
                    obs = self.running_mean_std(obs)
                    #because rn, proprio contains obs_buf 
                if self.normalize_proprio_hist:
                    proprio_hist = self.proprio_mean_std(proprio_hist)

                if self.normalize_pc:
                    pc_hist = self.pc_mean_std(pc_hist)

                assert not torch.any(obs.isnan())

                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'proprio_buf': proprio_hist, 
                    'pc_buf': pc_hist,
                    'action_buf': action_hist,
                    'attn_mask': attn_mask, 
                    'timesteps': timesteps
                }

                if self.num_gpus > 1:
                    res_dict = self.ddp_model(batch_dict)
                else:
                    res_dict = self.model(batch_dict)
              
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)

                a_loss, c_loss, entropy = [torch.mean(loss) for loss in [a_loss, c_loss, entropy]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef


                if self.gradient_steps < self.warmup_critic_gradsteps:
                    # Need the zeros, or something goes wrong! TODO: Fix this!
                    loss = 0*a_loss +  c_loss * self.critic_coef - entropy * 0

                else:
                    loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef

                self.optimizer.zero_grad()


                assert not torch.isnan(loss).any()

                loss.backward()

                self.gradient_steps += 1 

                if self.truncate_grads:
                    if self.num_gpus > 1:
                        torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss.detach())
                c_losses.append(c_loss.detach())
                ep_kls.append(kl)
                entropies.append(entropy.detach())
                self.replay_buffer.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr

            kls.append(av_kls)


        self.rl_train_time += (time.time() - _t)
        # print("Rl train time" , (time.time() - _t))

        train_info = {
            'losses/actor_loss' : ptu.to_numpy(torch.mean(torch.stack(a_losses))),
            'losses/critic_loss' : ptu.to_numpy(torch.mean(torch.stack(c_losses))),
            'losses/policy_entropy' : ptu.to_numpy(torch.mean(torch.stack(entropies))),
            'losses/kl_old-vs-new_policy' : ptu.to_numpy(torch.mean(torch.stack(kls))),
            'training/std' : self.model.logstd.exp().mean().cpu().detach().numpy(),
            'optimizer/learning_rate' : self.last_lr,
        }

        self.extra_info.update(train_info)
        return 

    

    def play_steps(self):

        for i in range(self.horizon_length):

            res_dict = self.get_action(self.obs)
            # collect o_t
            self.replay_buffer.update_data('obses', i, self.obs['obs'])
            
            if self.env.enable_proprio_history:
                self.replay_buffer.update_data('proprio_buf', i, self.obs['proprio_buf'])
                
            if self.env.enable_point_cloud:
                self.replay_buffer.update_data('pc_buf', i, self.obs['pc_buf'])
            
            if self.env.enable_action_history:
                self.replay_buffer.update_data('action_buf', i, self.obs['action_buf'])
            
            if self.env.enable_attn_mask:
                self.replay_buffer.update_data('attn_mask',i,self.obs['attn_mask'])
                self.replay_buffer.update_data('timesteps',i,self.obs['timesteps'])
                

            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.replay_buffer.update_data(k, i, res_dict[k])
        
            # do env step
            #actions = torch.clamp(res_dict['actions'], -1.0, 1.0) 
            """
                Policy Specific Decision: Beware of the action space
            """
            actions = torch.clamp(res_dict["actions"],-1.0,1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.replay_buffer.update_data('dones', i, self.dones)
            shaped_rewards = 0.1 * rewards.clone()  #to scale rewards for numerical stability  

            #are timeouts in my infos?
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.replay_buffer.update_data('rewards', i, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.get_action(self.obs)
        last_values = res_dict['values'] #last_values computed using the critic.

        #add to log
        self.extra_info.update(self.env.linearize_info(infos))

        self.cur_success = self.extra_info['successes']

        self.agent_steps += self.batch_size
        self.replay_buffer.compute_return(last_values, self.gamma, self.tau)
        self.replay_buffer.prepare_training()

        returns = self.replay_buffer.data_dict['returns']
        values = self.replay_buffer.data_dict['values']

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)    
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
            
        self.replay_buffer.data_dict['values'] = values
        self.replay_buffer.data_dict['returns'] = returns
    

    def play_steps_efficient_buffer(self):

        for i in range(self.horizon_length):

            # res_dict = self.get_action(self.obs)

            # collect o_t
            self.replay_buffer.update_data('obses', i, self.obs['obs'][:,])
            
            if self.env.enable_proprio_history:
                self.replay_buffer.update_data('proprio_buf', i, self.obs['proprio_buf'][:,-1])
                
            if self.env.enable_point_cloud:
                self.replay_buffer.update_data('pc_buf', i, self.obs['pc_buf'][:,-1])
            
            if self.env.enable_action_history:
                self.replay_buffer.update_data('action_buf', i, self.obs['action_buf'][:,-1])
            
            self.replay_buffer.update_data('env_iter', i, self.obs['env_id'][:,-1])

            # if self.env.enable_attn_mask:
            #     self.replay_buffer.update_data('attn_mask',i,self.obs['attn_mask'])
            #     self.replay_buffer.update_data('timesteps',i,self.obs['timesteps'])
            
            res_dict = self.get_action(self.obs)

            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.replay_buffer.update_data(k, i, res_dict[k])
        
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0) 

            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.replay_buffer.update_data('dones', i, self.dones)
            shaped_rewards = 0.1 * rewards.clone()  #to scale rewards for numerical stability  

            #are timeouts in my infos?
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.replay_buffer.update_data('rewards', i, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.get_action(self.obs)
        last_values = res_dict['values'] #last_values computed using the critic.

        #add to log
        self.extra_info.update(self.env.linearize_info(infos))

        self.cur_success = self.extra_info['successes']

        self.agent_steps += self.batch_size
        self.replay_buffer.compute_return(last_values, self.gamma, self.tau)
        self.replay_buffer.prepare_training()

        returns = self.replay_buffer.data_dict['returns']
        values = self.replay_buffer.data_dict['values']

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)    
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
            
        self.replay_buffer.data_dict['values'] = values
        self.replay_buffer.data_dict['returns'] = returns


    def test(self, visualise=True, name='PPOFinetune'):
        self.set_eval()
        obs_dict = self.env.reset()
        j = 0

        if visualise:
            all_rewards = [0.0,]*self.env.num_envs
            all_rewards = np.array(all_rewards)
            all_successes = np.array([0.0,]*self.env.num_envs)
        
        #once you fail in the test, you fail. That env is removed from metrics for success calculation
        envs_done = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        while j < 600:
            if self.normalize_input:
                        processed_obs = self.running_mean_std(obs_dict['obs'])
            else:
                processed_obs = obs_dict['obs']
            
            if self.normalize_proprio_hist:
                processed_proprio = self.proprio_mean_std(obs_dict['proprio_buf'].clone())
            else:
                processed_proprio = obs_dict['proprio_buf'].clone() #POLICY SPECIFIC SCALING: BEWARE OF THE ACTION SPACE

            if self.normalize_pc:
                processed_pc_hist = self.pc_mean_std(obs_dict['pc_buf'].clone())
            else:
                processed_pc_hist = obs_dict['pc_buf'].clone()

            input_dict = {
                "obs": processed_obs,
                "pc_buf": processed_pc_hist,
                "proprio_buf": processed_proprio,
                "action_buf": obs_dict['action_buf'],
                "attn_mask": obs_dict['attn_mask'],
                "timesteps": obs_dict['timesteps']
            }
            
            mu = self.model.infer_action(input_dict)            
            mu = torch.clamp(mu, -1.0, 1.0)
            next_obs_dict, r, done, info = self.env.step(mu)

            obs_dict = next_obs_dict
            
            if visualise:
                all_rewards += ptu.to_numpy(r)
                all_successes[~envs_done.detach().cpu().numpy()] += ptu.to_numpy(self.env.is_success[~envs_done])
            
            envs_done = torch.logical_or(envs_done, done)
            if self.logger is not None:

                self.logger.log_dict(self.env.linearize_info(info),j)

            j += 1 
        
        if visualise:
            from utils.misc import compile_per_asset
            compile_per_asset(self.env,all_rewards,all_successes, name=name)


        if self.log_video:
            video_frames = self.env.stop_video_recording()
            fps = int(1/(self.env.control_freq_inv * self.env.sim_params.dt))
            cprint(f"Logging video. Num frames {len(video_frames)}", 'green', attrs=['bold'])
            self.logger.log_video(video_frames, name="Videos", fps=fps, step=j)


# def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):

#     c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
#     c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
#     c3 = -1.0 / 2.0
#     kl = c1 + c2 + c3
#     kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
#     return kl.mean()


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):

    c1 = torch.log(p1_sigma/p0_sigma)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()

# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008, min_lr=1e-6, max_lr=1e-2):
        super().__init__()
        self.min_lr = min_lr 
        self.max_lr = max_lr 
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
