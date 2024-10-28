import numpy as np

import torch
import time
from torch.utils.data import DataLoader
import os 
from datetime import datetime 
import wandb 
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from termcolor import cprint 
class Trainer:

    def __init__(self, 
                 model, 
                 train_dataset, 
                 collate_fn, 
                 optimizer, 
                 loss_fn, 
                 model_save_dir, 
                 val_dataset=None, 
                 config=None, 
                 scheduler=None, 
                 eval_fns=None, 
                 logger=None, 
                 rank=0, 
                 world_size=1, 
                 device='cuda'):

        self.model = model
        self.optimizer = optimizer
        self.batch_size = config.pretrain.training.batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = model_save_dir
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.logger = logger
        self.saved_model_number = 0
        self.full_autoregressive = config.pretrain.model.full_autoregressive
        self.action_input = config.pretrain.model.action_input
        self.modality_aligned = config.pretrain.training.modality_aligned
        self.time_shift = config.pretrain.training.time_shift
        self.add_proprio_noise = config.pretrain.training.add_proprio_noise
        self.add_action_noise = config.pretrain.training.add_action_noise
        self.add_data_driven_noise = config.pretrain.training.add_data_driven_noise
        num_workers = 16 #config.pretrain.training.num_workers
        self.use_pc_loss = config.pretrain.training.use_pc_loss
        self.use_proprio_loss = config.pretrain.training.use_proprio_loss
        self.log_freq = config.pretrain.training.log_freq
        self.noise_arm = config.pretrain.training.noise_arm
        self.noise_hand = config.pretrain.training.noise_hand
        self.model_save_freq = config.pretrain.training.model_save_freq
        assert self.time_shift >= 0, "Cannot have negative time shift"
        assert self.time_shift < self.train_dataset.ctx-1, "Time shift cannot be larger than the context length"

        self.data_driven_noise = {'action': {'mean': np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.0, 0.07, 0.08, 0.05, 0.0, 0.1, 0.04, 0.06, 0.0, 0.04, 0.05, 0.05, 0.0]),
                                              'max': np.array([0.09, 0.07, 0.04, 0.11, 0.05, 0.13, 0.16, 0.28, 0.17, 0.23, 0.15, 0.31, 0.35, 0.15, 0.13, 0.43, 0.2, 0.37, 0.08, 0.25, 0.2, 0.22, 0.12]),
                                              'min': np.array([6.09e-06, 1.15e-04, 2.36e-03, 3.52e-05, 1.76e-05, 2.75e-05, 7.07e-05, 2.02e-04, 4.86e-04, 9.61e-05, 0.00e+00, 6.21e-04, 6.67e-05, 1.23e-03, 1.79e-06, 6.07e-04, 9.72e-05, 2.49e-04, 2.21e-06, 1.28e-04, 9.08e-04, 4.32e-04, 8.34e-07]),
                                              'std': np.array([0.02, 0.02, 0.01, 0.02, 0.01, 0.03, 0.04, 0.06, 0.04, 0.04, 0.02, 0.08, 0.09, 0.04, 0.02, 0.08, 0.04, 0.07, 0.01, 0.05, 0.05, 0.05, 0.01])}
        }

        for k in self.data_driven_noise:
            for kk in self.data_driven_noise[k]:
                self.data_driven_noise[k][kk] = torch.tensor(self.data_driven_noise[k][kk], dtype=torch.float32).to(device)


        if self.modality_aligned:
            self.proprio_shift = self.time_shift + 1
            self.action_shift = self.time_shift + 1
            self.pc_shift = self.time_shift + 1
        else:
            self.proprio_shift = self.time_shift + 1
            self.action_shift = self.time_shift
            self.pc_shift = self.time_shift

        # create a dataloader
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           num_workers=num_workers,
                                           shuffle=True, 
                                           collate_fn=self.collate_fn,
                                           prefetch_factor=8)
        
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=num_workers,
                                            shuffle=False, collate_fn=self.collate_fn)
        else:
            self.val_dataloader = None 

        self.start_time = time.time()

    def train_epoch(self, iter_num=0, print_logs=False):

        train_losses = []
        if self.full_autoregressive:
            train_losses_pc = []
            train_losses_next_proprio = []
            train_losses_action = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        
        t = time.time()
        for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):
            
            
            proprio, object_pc, actions, timesteps, attention_mask = batch
            batch = proprio.to(self.train_dataset.device), object_pc.to(self.train_dataset.device), \
                actions.to(self.train_dataset.device), timesteps.to(self.train_dataset.device), \
                attention_mask.to(self.train_dataset.device) if attention_mask is not None else None

        
            train_loss = self.train_step(batch)
            t = time.time()

            if self.full_autoregressive:
                
                if self.use_pc_loss and self.action_input:
                    train_losses_pc.append(train_loss['pc'])
            
            train_losses.append(train_loss['loss'])
        

            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.logger is not None and  i % self.log_freq == 0:
                logs['time/training'] = time.time() - train_start
                logs['time/total'] = time.time() - self.start_time
                logs['optimizer/lr'] = self.optimizer.param_groups[0]['lr']
                global_step = iter_num * len(self.train_dataloader) + i
                self.logger.log_dict(logs, global_step)
                if self.full_autoregressive:
                    logs['training/train_loss_mean'] = np.mean(train_losses)
                    logs['training/train_loss_std'] = np.std(train_losses)
                    if self.use_pc_loss and self.action_input:
                        logs['training/train_loss_pc_mean'] = np.mean(train_losses_pc)
                        logs['training/train_loss_pc_std'] = np.std(train_losses_pc)
                    if self.use_proprio_loss:
                        logs['training/train_loss_next_proprio_mean'] = np.mean(train_losses_next_proprio)
                        logs['training/train_loss_next_proprio_std'] = np.std(train_losses_next_proprio)
                else:
                    logs['training/train_loss_mean'] = np.mean(train_losses)
                    logs['training/train_loss_std'] = np.std(train_losses)

            if self.save_dir is not None and i % self.model_save_freq == 0:
                global_step = iter_num * len(self.train_dataloader) + i
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_step_{global_step}.pt'))
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'last.pt'))
                self.saved_model_number += 1

            if print_logs and i % self.log_freq == 0:
                for k in self.diagnostics:
                    logs[k] = self.diagnostics[k]
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
            

        return logs
    
    def eval_epoch(self, iter_num=0, print_logs=False):

        val_losses = []
        if self.full_autoregressive:
            val_losses_pc = []
            val_losses_next_proprio = []
            val_losses_action = []
        logs = dict()

        self.model.eval()
        
        for i, batch in enumerate(tqdm.tqdm(self.val_dataloader)):

            proprio, object_pc, actions, timesteps, attention_mask = batch
            batch = proprio.to(self.train_dataset.device), object_pc.to(self.train_dataset.device), \
                actions.to(self.train_dataset.device), timesteps.to(self.train_dataset.device), \
                attention_mask.to(self.train_dataset.device) if attention_mask is not None else None
            val_loss = self.validate_step(batch)

            if self.full_autoregressive:
                if self.use_pc_loss and self.action_input:
                    val_losses_pc.append(val_loss['pc'])
                val_losses.append(val_loss['loss'])
            else:   
                val_losses.append(val_loss)


        if self.logger:
            if self.full_autoregressive:
                logs['training/val_loss_full_mean'] = np.mean(val_losses)
                logs['training/val_loss_full_std'] = np.std(val_losses)
                if self.use_pc_loss and self.action_input:
                    logs['training/val_loss_pc_mean'] = np.mean(val_losses_pc)
                    logs['training/val_loss_pc_std'] = np.std(val_losses_pc)
                if self.use_proprio_loss:
                    logs['training/val_loss_next_proprio_mean'] = np.mean(val_losses_next_proprio)
                    logs['training/val_loss_next_proprio_std'] = np.std(val_losses_next_proprio)
            else:
                logs['training/val_loss_mean'] = np.mean(val_losses)
                logs['training/val_loss_std'] = np.std(val_losses)
            self.logger.log_dict(logs, iter_num*len(self.val_dataloader))

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'Validation {k}: {v}')

        return logs
    
    def  validate_step(self,batch):
        raise NotImplementedError

    def train_step(self,batch):
        raise NotImplementedError

class MultiGPUTrainer:

    def __init__(self, 
                 model, 
                 train_dataset, 
                 collate_fn, 
                 loss_fn, 
                 model_save_dir, 
                 val_dataset=None, 
                 config=None, 
                 scheduler=None, 
                 eval_fns=None, 
                 logger=None, 
                 rank=0, 
                 world_size=1, 
                 device='cuda'):

        self.model = model
        self.rank = rank 
        self.world_size = world_size 
        if self.world_size > 1:
            self.device = f'cuda:{self.rank}'
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank) 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.pretrain.training.lr,weight_decay=config.pretrain.training.weight_decay)
        self.batch_size = config.pretrain.training.batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = model_save_dir
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.logger = logger
        self.saved_model_number = 0
        self.full_autoregressive = config.pretrain.model.full_autoregressive
        self.action_input = config.pretrain.model.action_input
        self.modality_aligned = config.pretrain.training.modality_aligned
        self.time_shift = config.pretrain.training.time_shift
        self.add_proprio_noise = config.pretrain.training.add_proprio_noise
        self.add_action_noise = config.pretrain.training.add_action_noise
        self.add_data_driven_noise = config.pretrain.training.add_data_driven_noise
        self.num_workers = config.pretrain.training.num_workers
        self.use_pc_loss = config.pretrain.training.use_pc_loss
        self.use_proprio_loss = config.pretrain.training.use_proprio_loss
        self.log_freq = config.pretrain.training.log_freq
        self.noise_arm = config.pretrain.training.noise_arm
        self.noise_hand = config.pretrain.training.noise_hand
        self.model_save_freq = config.pretrain.training.model_save_freq
        assert self.time_shift >= 0, "Cannot have negative time shift"
        assert self.time_shift < self.train_dataset.ctx-1, "Time shift cannot be larger than the context length"

        self.data_driven_noise = {'action': {'mean': np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.04, 0.0, 0.07, 0.08, 0.05, 0.0, 0.1, 0.04, 0.06, 0.0, 0.04, 0.05, 0.05, 0.0]),
                                              'max': np.array([0.09, 0.07, 0.04, 0.11, 0.05, 0.13, 0.16, 0.28, 0.17, 0.23, 0.15, 0.31, 0.35, 0.15, 0.13, 0.43, 0.2, 0.37, 0.08, 0.25, 0.2, 0.22, 0.12]),
                                              'min': np.array([6.09e-06, 1.15e-04, 2.36e-03, 3.52e-05, 1.76e-05, 2.75e-05, 7.07e-05, 2.02e-04, 4.86e-04, 9.61e-05, 0.00e+00, 6.21e-04, 6.67e-05, 1.23e-03, 1.79e-06, 6.07e-04, 9.72e-05, 2.49e-04, 2.21e-06, 1.28e-04, 9.08e-04, 4.32e-04, 8.34e-07]),
                                              'std': np.array([0.02, 0.02, 0.01, 0.02, 0.01, 0.03, 0.04, 0.06, 0.04, 0.04, 0.02, 0.08, 0.09, 0.04, 0.02, 0.08, 0.04, 0.07, 0.01, 0.05, 0.05, 0.05, 0.01])}
        }

        for k in self.data_driven_noise:
            for kk in self.data_driven_noise[k]:
                self.data_driven_noise[k][kk] = torch.tensor(self.data_driven_noise[k][kk], dtype=torch.float32).to(device)


        if self.modality_aligned:
            self.proprio_shift = self.time_shift + 1
            self.action_shift = self.time_shift + 1
            self.pc_shift = self.time_shift + 1
        else:
            self.proprio_shift = self.time_shift + 1
            self.action_shift = self.time_shift
            self.pc_shift = self.time_shift
        
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, sampler=sampler)
            if self.val_dataset is not None:
                sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank)
                self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,collate_fn=self.collate_fn, sampler=sampler)
        else:
            # create a dataloader
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                            shuffle=True, collate_fn=self.collate_fn)

            if self.val_dataset is not None:
                self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                shuffle=False, collate_fn=self.collate_fn)

        self.start_time = time.time()

    def train_epoch(self, iter_num=0, print_logs=False):

        train_losses = []
        if self.full_autoregressive:
            train_losses_pc = []
            train_losses_next_proprio = []
            train_losses_action = []
        logs = dict()

        train_start = time.time()

        self.model.train()

        for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):

            proprio, object_pc, actions, timesteps, attention_mask = batch
            batch = proprio.to(self.train_dataset.device), object_pc.to(self.train_dataset.device), \
                actions.to(self.train_dataset.device), timesteps.to(self.train_dataset.device), \
                attention_mask.to(self.train_dataset.device) if attention_mask is not None else None

            train_loss = self.train_step(batch)

            if self.full_autoregressive:
                if self.use_pc_loss and self.action_input:
                    train_losses_pc.append(train_loss['pc'])
                train_losses_action.append(train_loss['action'])
                train_losses.append(train_loss['full'])
            else:
                train_losses.append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.logger is not None and  i % self.log_freq == 0 and (self.world_size == 1 or self.rank==0):
                logs['time/training'] = time.time() - train_start
                logs['time/total'] = time.time() - self.start_time
                logs['optimizer/lr'] = self.optimizer.param_groups[0]['lr']
                global_step = iter_num * len(self.train_dataloader) + i
                self.logger.log_dict(logs, global_step)
                if self.full_autoregressive:
                    logs['training/train_loss_mean'] = np.mean(train_losses)
                    logs['training/train_loss_std'] = np.std(train_losses)
                    logs['training/train_loss_action_mean'] = np.mean(train_losses_action)
                    logs['training/train_loss_action_std'] = np.std(train_losses_action)
                    if self.use_pc_loss and self.action_input:
                        logs['training/train_loss_pc_mean'] = np.mean(train_losses_pc)
                        logs['training/train_loss_pc_std'] = np.std(train_losses_pc)
                    if self.use_proprio_loss:
                        logs['training/train_loss_next_proprio_mean'] = np.mean(train_losses_next_proprio)
                        logs['training/train_loss_next_proprio_std'] = np.std(train_losses_next_proprio)
                else:
                    logs['training/train_loss_mean'] = np.mean(train_losses)
                    logs['training/train_loss_std'] = np.std(train_losses)

            if self.save_dir is not None and i % self.model_save_freq == 0 and (self.world_size == 1 or self.rank==0):
                global_step = iter_num * len(self.train_dataloader) + i
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_step_{global_step}.pt'))
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'last.pt'))
                self.saved_model_number += 1

            if print_logs and i % self.log_freq == 0 and (self.world_size == 1 or self.rank==0):
                for k in self.diagnostics:
                    logs[k] = self.diagnostics[k]
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
            
            # if self.world_size > 1:
            #     torch.distributed.barrier()

        return logs
    
    def eval_epoch(self, iter_num=0, print_logs=False):

        val_losses = []
        if self.full_autoregressive:
            val_losses_pc = []
            val_losses_next_proprio = []
            val_losses_action = []
        logs = dict()

        self.model.eval()
        
        for i, batch in enumerate(tqdm.tqdm(self.val_dataloader)):

            proprio, object_pc, actions, timesteps, attention_mask = batch
            batch = proprio.to(self.train_dataset.device), object_pc.to(self.train_dataset.device), \
                actions.to(self.train_dataset.device), timesteps.to(self.train_dataset.device), \
                attention_mask.to(self.train_dataset.device) if attention_mask is not None else None
            val_loss = self.validate_step(batch)

            if self.full_autoregressive:
                if self.use_pc_loss and self.action_input:
                    val_losses_pc.append(val_loss['pc'])
                val_losses_next_proprio.append(val_loss['next_proprio'])
                val_losses_action.append(val_loss['action'])
                val_losses.append(val_loss['full'])
            else:   
                val_losses.append(val_loss)


        if self.logger:
            if self.full_autoregressive:
                logs['training/val_loss_action_mean'] = np.mean(val_losses_action)
                logs['training/val_loss_action_std'] = np.std(val_losses_action)
                logs['training/val_loss_full_mean'] = np.mean(val_losses)
                logs['training/val_loss_full_std'] = np.std(val_losses)
                if self.use_pc_loss and self.action_input:
                    logs['training/val_loss_pc_mean'] = np.mean(val_losses_pc)
                    logs['training/val_loss_pc_std'] = np.std(val_losses_pc)
                if self.use_proprio_loss:
                    logs['training/val_loss_next_proprio_mean'] = np.mean(val_losses_next_proprio)
                    logs['training/val_loss_next_proprio_std'] = np.std(val_losses_next_proprio)
            else:
                logs['training/val_loss_mean'] = np.mean(val_losses)
                logs['training/val_loss_std'] = np.std(val_losses)
            self.logger.log_dict(logs, iter_num*len(self.val_dataloader))

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'Validation {k}: {v}')

        return logs
    
    def  validate_step(self,batch):
        raise NotImplementedError

    def train_step(self,batch):
        raise NotImplementedError


class RobotTrainer(Trainer):

    def train_step(self,batch):
        
        proprio, depth, actions, timesteps, attention_mask = batch 

        action_target = torch.clone(actions)
        proprio_copy = torch.clone(proprio)
        
        if self.add_proprio_noise:
            if self.add_data_driven_noise:
                noise = torch.normal(mean=torch.zeros_like(proprio), std=self.data_driven_noise['action']['mean'])
            else:
                noise = torch.zeros_like(proprio)
                noise[...,:7] = torch.randn_like(proprio[...,:7])*self.noise_arm
                noise[...,7:] = torch.randn_like(proprio[...,7:])*self.noise_hand
            proprio = proprio + noise
        if self.add_action_noise:
            if self.add_data_driven_noise:
                noise = torch.normal(mean=torch.zeros_like(actions), std=self.data_driven_noise['action']['mean'])
            else:
                noise = torch.zeros_like(actions)
                noise[...,:7] = torch.randn_like(actions[...,:7])*self.noise_arm
                noise[...,7:] = torch.randn_like(actions[...,7:])*self.noise_hand
            actions = actions + noise


        if self.full_autoregressive:
            if self.action_input:
                pred_dict, _ = self.model.forward(
                    proprio, depth, actions, timesteps, attention_mask=attention_mask,
                )
                pc_preds = pred_dict['pc']
                pc_dim = pc_preds.shape[2]
            else:
                pred_dict, _ = self.model.forward(
                    proprio, depth, timesteps=timesteps, attention_mask=attention_mask,
                )
                
        else:
            pred_dict, _ = self.model.forward(
                proprio, object_pc, actions, timesteps, attention_mask=attention_mask,
            )
        
        action_preds = pred_dict["action"]



        act_dim = action_preds.shape[2]

        if self.action_shift > 0:
            action_target = torch.roll(action_target, shifts=-self.action_shift, dims=1)
            action_target = action_target[:, :-self.action_shift, :]
            action_preds = action_preds[:, :-self.action_shift, :]

        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.full_autoregressive:
            if self.action_input: 
                object_pc = object_pc.reshape(object_pc.shape[0], -1, pc_dim)
                if self.pc_shift > 0:
                    object_pc = object_pc[:, :-self.pc_shift, :]
                    pc_preds = pc_preds[:, :-self.pc_shift, :]
                if attention_mask is not None:
                    pc_preds = pc_preds.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                    object_pc = object_pc.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]

                loss_pc = self.loss_fn(pc_preds, object_pc)


            if attention_mask is not None:
                if self.use_proprio_loss:
                    # to compute the target, move the proprio to the next time step
                    next_proprio_target = torch.roll(proprio_copy, shifts=-(self.proprio_shift), dims=1)
                    # Skip the last time step
                    next_proprio_target = next_proprio_target[:, :-(self.proprio_shift), :]
                    next_proprio = next_proprio[:, :-(self.proprio_shift), :]
                    next_proprio_target = next_proprio_target.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
                    next_proprio = next_proprio.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
                    loss_next_proprio = self.loss_fn(next_proprio, next_proprio_target)

        loss_action = self.loss_fn(action_preds, action_target)

        if self.full_autoregressive:
            loss = loss_action
            if self.use_proprio_loss:
                loss += loss_next_proprio
            if self.use_pc_loss and self.action_input:
                loss += loss_pc
        else:
            loss = loss_action

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss_action.detach().cpu().item()
            if self.full_autoregressive:
                if self.use_proprio_loss:
                    self.diagnostics['training/next_proprio_error'] = loss_next_proprio.detach().cpu().item()
                if self.use_pc_loss and self.action_input:
                    self.diagnostics['training/pc_error'] = loss_pc.detach().cpu().item()
        
        if self.full_autoregressive:
            return_dict = {'loss': loss.detach().cpu().item()}
            if self.use_pc_loss and self.action_input:
                return_dict['pc'] = loss_pc.detach().cpu().item()
            return return_dict
        else:
            return loss.detach().cpu().item()
    
    @torch.no_grad()
    def validate_step(self, batch):
            
        proprio, depth, actions, timesteps, attention_mask = batch 
        action_target = torch.clone(actions)

        if self.full_autoregressive:
            if self.action_input:
                pred_dict, _ = self.model.forward(
                    proprio, depth, actions, timesteps, attention_mask=attention_mask,
                )
                pc_preds = pred_dict['pc']
                pc_dim = pc_preds.shape[2]
            else:
                pred_dict, _ = self.model.forward(
                    proprio, depth, timesteps=timesteps, attention_mask=attention_mask,
                )
            action_preds = pred_dict['action']
            next_proprio = pred_dict['next_proprio']
            next_proprio_dim = next_proprio.shape[2]
        else:
            action_preds, _ = self.model.forward(
                proprio, depth, actions, timesteps, attention_mask=attention_mask,
            )
    
        act_dim = action_preds.shape[2]

        if self.time_shift > 0:
            action_target = torch.roll(action_target, shifts=-self.time_shift, dims=1)
            action_target = action_target[:, :-self.time_shift, :]
            action_preds = action_preds[:, :-self.time_shift, :]
    
        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.full_autoregressive:
            if self.action_input:
                object_pc = object_pc.reshape(object_pc.shape[0], -1, pc_dim)
                if self.time_shift > 0:
                    object_pc = object_pc[:, :-self.time_shift, :]
                    pc_preds = pc_preds[:, :-self.time_shift, :]
                if attention_mask is not None:
                    pc_preds = pc_preds.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                    object_pc = object_pc.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                loss_pc = self.loss_fn(pc_preds, object_pc)

            # to compute the target, move the proprio to the next time step
            next_proprio_target = torch.roll(proprio, shifts=-(self.time_shift+1), dims=1)
            # Skip the last time step
            next_proprio_target = next_proprio_target[:, :-(self.time_shift+1), :]
            next_proprio = next_proprio[:, :-(self.time_shift+1), :]

            if attention_mask is not None:
                next_proprio_target = next_proprio_target.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
                next_proprio = next_proprio.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]

            loss_next_proprio = self.loss_fn(next_proprio,next_proprio_target)

        loss_action = self.loss_fn(action_preds,action_target)

        if self.full_autoregressive:
            loss = loss_action
            if self.use_proprio_loss:
                loss += loss_next_proprio
            if self.use_pc_loss and self.action_input:
                loss += loss_pc
            loss = self.loss_fn(action_preds,action_target)
            return_dict = {'loss': loss_action.detach().cpu().item()}
            if self.use_pc_loss and self.action_input:
                return_dict['pc'] = loss_pc.detach().cpu().item()
            return return_dict
        else:
            return loss_action.detach().cpu().item()



class RobotTrainerWithDepth(Trainer):

    def train_step(self,batch):
        
        proprio, depth, actions, timesteps, attention_mask = batch 

        action_target = torch.clone(actions)
        proprio_copy = torch.clone(proprio)
        
        if self.add_proprio_noise:
            if self.add_data_driven_noise:
                noise = torch.normal(mean=torch.zeros_like(proprio), std=self.data_driven_noise['action']['mean'])
            else:
                noise = torch.zeros_like(proprio)
                noise[...,:7] = torch.randn_like(proprio[...,:7])*self.noise_arm
                noise[...,7:] = torch.randn_like(proprio[...,7:])*self.noise_hand
            proprio = proprio + noise

        if self.add_action_noise:
            if self.add_data_driven_noise:
                noise = torch.normal(mean=torch.zeros_like(actions), std=self.data_driven_noise['action']['mean'])
            else:
                noise = torch.zeros_like(actions)
                noise[...,:7] = torch.randn_like(actions[...,:7])*self.noise_arm
                noise[...,7:] = torch.randn_like(actions[...,7:])*self.noise_hand
            actions = actions + noise

        if self.full_autoregressive:
            if self.action_input:
                pred_dict, _ = self.model.forward(
                    proprio, depth, actions, timesteps, attention_mask=attention_mask,
                )
                pc_preds = pred_dict['pc']
                pc_dim = pc_preds.shape[2]
            else:
                action_preds, _ = self.model.forward(
                    proprio, depth, timesteps=timesteps, attention_mask=attention_mask,
                )
        else:
            action_preds, _ = self.model.forward(
                proprio, object_pc, actions, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]

        if self.action_shift > 0:
            action_target = torch.roll(action_target, shifts=-self.action_shift, dims=1)
            action_target = action_target[:, :-self.action_shift, :]
            action_preds = action_preds[:, :-self.action_shift, :]

        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.full_autoregressive:
            if self.action_input: 
                object_pc = object_pc.reshape(object_pc.shape[0], -1, pc_dim)
                if self.pc_shift > 0:
                    object_pc = object_pc[:, :-self.pc_shift, :]
                    pc_preds = pc_preds[:, :-self.pc_shift, :]
                if attention_mask is not None:
                    pc_preds = pc_preds.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                    object_pc = object_pc.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]

                loss_pc = self.loss_fn(pc_preds, object_pc)


            # if attention_mask is not None:
            #     # to compute the target, move the proprio to the next time step
            #     next_proprio_target = torch.roll(proprio_copy, shifts=-(self.proprio_shift), dims=1)
            #     # Skip the last time step
            #     next_proprio_target = next_proprio_target[:, :-(self.proprio_shift), :]
            #     next_proprio = next_proprio[:, :-(self.proprio_shift), :]
            #     next_proprio_target = next_proprio_target.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
            #     next_proprio = next_proprio.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]

            # loss_next_proprio = self.loss_fn(next_proprio, next_proprio_target)

        loss_action = self.loss_fn(action_preds, action_target)

        if self.full_autoregressive:
            loss = loss_action
            if self.use_proprio_loss:
                loss += loss_next_proprio
            if self.use_pc_loss and self.action_input:
                loss += loss_pc
        else:
            loss = loss_action

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss_action.detach().cpu().item()
            if self.full_autoregressive:
                if self.use_proprio_loss:
                    self.diagnostics['training/next_proprio_error'] = loss_next_proprio.detach().cpu().item()
                if self.use_pc_loss and self.action_input:
                    self.diagnostics['training/pc_error'] = loss_pc.detach().cpu().item()
        
        if self.full_autoregressive:
            return_dict = {'action': loss_action.detach().cpu().item(),
                           'full': loss.detach().cpu().item()}
            if self.use_pc_loss and self.action_input:
                return_dict['pc'] = loss_pc.detach().cpu().item()
            return return_dict
        else:
            return loss.detach().cpu().item()
    
    @torch.no_grad()
    def validate_step(self, batch):
            
        proprio, depth, actions, timesteps, attention_mask = batch 
        action_target = torch.clone(actions)

        if self.full_autoregressive:
            if self.action_input:
                pred_dict, _ = self.model.forward(
                    proprio, depth, actions, timesteps, attention_mask=attention_mask,
                )
                pc_preds = pred_dict['pc']
                pc_dim = pc_preds.shape[2]
            else:
                action_preds, _ = self.model.forward(
                    proprio, depth, timesteps=timesteps, attention_mask=attention_mask,
                )
            # action_preds = pred_dict['action']
            # next_proprio = pred_dict['next_proprio']
            # next_proprio_dim = next_proprio.shape[2]
        else:
            action_preds, _ = self.model.forward(
                proprio, depth, actions, timesteps, attention_mask=attention_mask,
            )
    
        act_dim = action_preds.shape[2]

        if self.time_shift > 0:
            action_target = torch.roll(action_target, shifts=-self.time_shift, dims=1)
            action_target = action_target[:, :-self.time_shift, :]
            action_preds = action_preds[:, :-self.time_shift, :]
    
        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.full_autoregressive:
            if self.action_input:
                object_pc = object_pc.reshape(object_pc.shape[0], -1, pc_dim)
                if self.time_shift > 0:
                    object_pc = object_pc[:, :-self.time_shift, :]
                    pc_preds = pc_preds[:, :-self.time_shift, :]
                if attention_mask is not None:
                    pc_preds = pc_preds.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                    object_pc = object_pc.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                loss_pc = self.loss_fn(pc_preds, object_pc)

            # to compute the target, move the proprio to the next time step
            next_proprio_target = torch.roll(proprio, shifts=-(self.time_shift+1), dims=1)
            # Skip the last time step
            next_proprio_target = next_proprio_target[:, :-(self.time_shift+1), :]
            next_proprio = next_proprio[:, :-(self.time_shift+1), :]

            if attention_mask is not None:
                next_proprio_target = next_proprio_target.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
                next_proprio = next_proprio.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]

            loss_next_proprio = self.loss_fn(next_proprio,next_proprio_target)

        loss_action = self.loss_fn(action_preds,action_target)

        if self.full_autoregressive:
            loss = loss_action
            if self.use_proprio_loss:
                loss += loss_next_proprio
            if self.use_pc_loss and self.action_input:
                loss += loss_pc
            loss = self.loss_fn(action_preds,action_target)
            return_dict = {'action': loss_action.detach().cpu().item(),
                           'next_proprio': loss_next_proprio.detach().cpu().item(),
                           'full': loss.detach().cpu().item()}
            if self.use_pc_loss and self.action_input:
                return_dict['pc'] = loss_pc.detach().cpu().item()
            return return_dict
        else:
            return loss_action.detach().cpu().item()




class MultiGPURobotTrainerWithDepth(MultiGPUTrainer):

    def train_step(self,batch):
        
        
        proprio, depth, actions, timesteps, attention_mask = batch 

        action_target = torch.clone(actions)
        proprio_copy = torch.clone(proprio)
        
        if self.add_proprio_noise:
            if self.add_data_driven_noise:
                noise = torch.normal(mean=torch.zeros_like(proprio), std=self.data_driven_noise['action']['mean'])
            else:
                noise = torch.zeros_like(proprio)
                noise[...,:7] = torch.randn_like(proprio[...,:7])*self.noise_arm
                noise[...,7:] = torch.randn_like(proprio[...,7:])*self.noise_hand
            proprio = proprio + noise

        if self.add_action_noise:
            if self.add_data_driven_noise:
                noise = torch.normal(mean=torch.zeros_like(actions), std=self.data_driven_noise['action']['mean'])
            else:
                noise = torch.zeros_like(actions)
                noise[...,:7] = torch.randn_like(actions[...,:7])*self.noise_arm
                noise[...,7:] = torch.randn_like(actions[...,7:])*self.noise_hand
            actions = actions + noise

        if self.full_autoregressive:
            if self.action_input:
                pred_dict, _ = self.model.forward(
                    proprio, depth, actions, timesteps, attention_mask=attention_mask,
                )
                pc_preds = pred_dict['pc']
                pc_dim = pc_preds.shape[2]
            else:
                action_preds, _ = self.model.forward(
                    proprio, depth, timesteps=timesteps, attention_mask=attention_mask,
                )
        else:
            action_preds, _ = self.model.forward(
                proprio, object_pc, actions, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]

        if self.action_shift > 0:
            action_target = torch.roll(action_target, shifts=-self.action_shift, dims=1)
            action_target = action_target[:, :-self.action_shift, :]
            action_preds = action_preds[:, :-self.action_shift, :]

        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.full_autoregressive:
            if self.action_input: 
                object_pc = object_pc.reshape(object_pc.shape[0], -1, pc_dim)
                if self.pc_shift > 0:
                    object_pc = object_pc[:, :-self.pc_shift, :]
                    pc_preds = pc_preds[:, :-self.pc_shift, :]
                if attention_mask is not None:
                    pc_preds = pc_preds.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                    object_pc = object_pc.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]

                loss_pc = self.loss_fn(pc_preds, object_pc)


            # if attention_mask is not None:
            #     # to compute the target, move the proprio to the next time step
            #     next_proprio_target = torch.roll(proprio_copy, shifts=-(self.proprio_shift), dims=1)
            #     # Skip the last time step
            #     next_proprio_target = next_proprio_target[:, :-(self.proprio_shift), :]
            #     next_proprio = next_proprio[:, :-(self.proprio_shift), :]
            #     next_proprio_target = next_proprio_target.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
            #     next_proprio = next_proprio.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]

            # loss_next_proprio = self.loss_fn(next_proprio, next_proprio_target)

        loss_action = self.loss_fn(action_preds, action_target)

        if self.full_autoregressive:
            loss = loss_action
            if self.use_proprio_loss:
                loss += loss_next_proprio
            if self.use_pc_loss and self.action_input:
                loss += loss_pc
        else:
            loss = loss_action

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss_action.detach().cpu().item()
            if self.full_autoregressive:
                if self.use_proprio_loss:
                    self.diagnostics['training/next_proprio_error'] = loss_next_proprio.detach().cpu().item()
                if self.use_pc_loss and self.action_input:
                    self.diagnostics['training/pc_error'] = loss_pc.detach().cpu().item()
        
        if self.full_autoregressive:
            return_dict = {'action': loss_action.detach().cpu().item(),
                           'full': loss.detach().cpu().item()}
            if self.use_pc_loss and self.action_input:
                return_dict['pc'] = loss_pc.detach().cpu().item()
            return return_dict
        else:
            return loss.detach().cpu().item()
    
    @torch.no_grad()
    def validate_step(self, batch):
            
        proprio, depth, actions, timesteps, attention_mask = batch 
        action_target = torch.clone(actions)

        if self.full_autoregressive:
            if self.action_input:
                pred_dict, _ = self.model.forward(
                    proprio, depth, actions, timesteps, attention_mask=attention_mask,
                )
                pc_preds = pred_dict['pc']
                pc_dim = pc_preds.shape[2]
            else:
                action_preds, _ = self.model.forward(
                    proprio, depth, timesteps=timesteps, attention_mask=attention_mask,
                )
            # action_preds = pred_dict['action']
            # next_proprio = pred_dict['next_proprio']
            # next_proprio_dim = next_proprio.shape[2]
        else:
            action_preds, _ = self.model.forward(
                proprio, depth, actions, timesteps, attention_mask=attention_mask,
            )
    
        act_dim = action_preds.shape[2]

        if self.time_shift > 0:
            action_target = torch.roll(action_target, shifts=-self.time_shift, dims=1)
            action_target = action_target[:, :-self.time_shift, :]
            action_preds = action_preds[:, :-self.time_shift, :]
    
        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.full_autoregressive:
            if self.action_input:
                object_pc = object_pc.reshape(object_pc.shape[0], -1, pc_dim)
                if self.time_shift > 0:
                    object_pc = object_pc[:, :-self.time_shift, :]
                    pc_preds = pc_preds[:, :-self.time_shift, :]
                if attention_mask is not None:
                    pc_preds = pc_preds.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                    object_pc = object_pc.reshape(-1, pc_dim)[attention_mask.reshape(-1) > 0]
                loss_pc = self.loss_fn(pc_preds, object_pc)

            # to compute the target, move the proprio to the next time step
            next_proprio_target = torch.roll(proprio, shifts=-(self.time_shift+1), dims=1)
            # Skip the last time step
            next_proprio_target = next_proprio_target[:, :-(self.time_shift+1), :]
            next_proprio = next_proprio[:, :-(self.time_shift+1), :]

            if attention_mask is not None:
                next_proprio_target = next_proprio_target.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]
                next_proprio = next_proprio.reshape(-1, next_proprio_dim)[attention_mask.reshape(-1) > 0]

            loss_next_proprio = self.loss_fn(next_proprio,next_proprio_target)

        loss_action = self.loss_fn(action_preds,action_target)

        if self.full_autoregressive:
            loss = loss_action
            if self.use_proprio_loss:
                loss += loss_next_proprio
            if self.use_pc_loss and self.action_input:
                loss += loss_pc
            loss = self.loss_fn(action_preds,action_target)
            return_dict = {'action': loss_action.detach().cpu().item(),
                           'next_proprio': loss_next_proprio.detach().cpu().item(),
                           'full': loss.detach().cpu().item()}
            if self.use_pc_loss and self.action_input:
                return_dict['pc'] = loss_pc.detach().cpu().item()
            return return_dict
        else:
            return loss_action.detach().cpu().item()

