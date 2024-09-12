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
class DepthTrainer:

    def __init__(self, 
                 model, 
                 collate_fn, 
                 optimizer, 
                 loss_fn, 
                 model_save_dir, 
                 train_dataloader,
                 val_dataset=None, 
                 config=None, 
                 scheduler=None, 
                 eval_fns=None, 
                 logger=None, 
                 rank=0, 
                 world_size=1, 
                 device='cuda'):

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.batch_size = config.pretrain.training.batch_size
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = model_save_dir
        self.rank = rank 
        self.world_size = world_size 
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.logger = logger
        self.saved_model_number = 0
        self.add_proprio_noise = config.pretrain.training.add_proprio_noise
        self.add_action_noise = config.pretrain.training.add_action_noise
        num_workers = config.pretrain.training.num_workers #add this to bash file 
        self.log_freq = config.pretrain.training.log_freq
        self.model_save_freq = config.pretrain.training.model_save_freq
        # create a dataloader
        self.train_dataloader = train_dataloader 

        self.start_time = time.time()

    def train_epoch(self, iter_num=0, print_logs=False):

        train_losses = []
        train_losses_action = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        
        for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):
            
            proprio, depth , actions, timesteps, attention_mask = batch
            batch = proprio.to(self.device), depth.to(self.device), \
                actions.to(self.device), timesteps.to(self.device), \
                attention_mask.to(self.device) if attention_mask is not None else None

            train_loss = self.train_step(batch)

            train_losses_action.append(train_loss['action'])
            train_losses.append(train_loss['full'])

            if self.scheduler is not None:
                self.scheduler.step()
            
            if self.logger is not None and  i % self.log_freq == 0:
                logs['time/training'] = time.time() - train_start
                logs['time/total'] = time.time() - self.start_time
                logs['optimizer/lr'] = self.optimizer.param_groups[0]['lr']
                global_step = iter_num * len(self.train_dataloader) + i
                self.logger.log_dict(logs, global_step)
                logs['training/train_loss_mean'] = np.mean(train_losses)
                logs['training/train_loss_std'] = np.std(train_losses)
                logs['training/train_loss_action_mean'] = np.mean(train_losses_action)
                logs['training/train_loss_action_std'] = np.std(train_losses_action)

            global_step = iter_num * len(self.train_dataloader) + i
            if self.save_dir is not None and global_step % self.model_save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_step_{global_step}.pt'))
                self.saved_model_number += 1
            
            #if self.save_dir is not None and global_step % self.model_save_freq == 0:
            #torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_step_{global_step}.pt'))

            if print_logs and i % self.log_freq == 0:
                for k in self.diagnostics:
                    logs[k] = self.diagnostics[k]
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')

        return logs
    
    def train_step(self,batch):
        
        proprio, depth, actions, timesteps, attention_mask = batch



        action_target = torch.clone(actions)
        
        if self.add_proprio_noise:
            noise = torch.zeros_like(proprio)
            noise[...,:7] = torch.randn_like(proprio[...,:7])*0.1 #self.noise_arm
            noise[...,7:] = torch.randn_like(proprio[...,7:])*0.1 #self.noise_hand
            proprio = proprio + noise


        action_preds, _ = self.model.forward(proprio,depth,timesteps,attention_mask)
       
        act_dim = action_preds.shape[2]

        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]


        loss_action = self.loss_fn(action_preds, action_target)

        loss = loss_action

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss_action.detach().cpu().item()
        
        return_dict = {'action': loss_action.detach().cpu().item(),
                        'full': loss.detach().cpu().item()
                        }
        
        return return_dict 

