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

class MultiGPUTrainer:

    def __init__(self, 
                 model, 
                 train_dataset, 
                 collate_fn, 
                 loss_fn, 
                 model_save_dir, 
                 rank,
                 world_size,
                 val_dataset=None, 
                 config=None, 
                 scheduler=None, 
                 eval_fns=None, 
                 logger=None, 
                 device='cuda'):

        self.model = model
        self.rank = rank 
        self.world_size = world_size 
        if self.world_size > 1:
            self.device = f'cuda:{self.rank}'
            self.model = self.model.to(self.device)
            self.ddp_model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)
            self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=config.pretrain.training.lr*config.num_gpus,weight_decay=config.pretrain.training.weight_decay)
        else:
            self.device = device 
            self.model = self.model.to(self.device)
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
        self.action_input = config.pretrain.model.action_input
        self.add_proprio_noise = config.pretrain.training.add_proprio_noise
        self.add_action_noise = config.pretrain.training.add_action_noise
        self.num_workers = config.pretrain.training.num_workers
        self.log_freq = config.pretrain.training.log_freq
        self.noise_arm = config.pretrain.training.noise_arm
        self.noise_hand = config.pretrain.training.noise_hand
        self.model_save_freq = config.pretrain.training.model_save_freq


        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
            self.train_dataloader = DataLoader(self.train_dataset, 
                                               batch_size=self.batch_size, 
                                               num_workers=self.num_workers, 
                                               collate_fn=self.collate_fn, 
                                               sampler=sampler)
            if self.val_dataset is not None:
                sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, 
                                                                          num_replicas=world_size, 
                                                                          rank=rank)
                
                self.val_dataloader = DataLoader(self.val_dataset, 
                                                 batch_size=self.batch_size, 
                                                 num_workers=self.num_workers,
                                                 collate_fn=self.collate_fn, 
                                                 sampler=sampler)
        else:
            # create a dataloader
            print('Creating dataloader')
            self.train_dataloader = DataLoader(self.train_dataset, 
                                               batch_size=self.batch_size, 
                                               num_workers=self.num_workers,
                                               shuffle=True, 
                                               collate_fn=self.collate_fn)

            if self.val_dataset is not None:
                self.val_dataloader = DataLoader(self.val_dataset, 
                                                 batch_size=self.batch_size, 
                                                 num_workers=self.num_workers,
                                                 shuffle=False, 
                                                 collate_fn=self.collate_fn)

        self.start_time = time.time()

    def train_epoch(self, iter_num=0, print_logs=False):

        train_losses, train_losses_action = [], []
        logs = dict()

        train_start = time.time()

        if self.world_size > 1:
            self.ddp_model.train()
        self.model.train()

        if self.world_size > 1:
            self.train_dataloader.sampler.set_epoch(iter_num)

        for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):

            proprio, depth, actions, timesteps, attention_mask = batch
            batch = proprio.to(self.device), depth.to(self.device), \
                actions.to(self.device), timesteps.to(self.device), \
                attention_mask.to(self.device) if attention_mask is not None else None


            print(self.device)
            train_loss = self.train_step(batch)

            print(train_loss)

            train_losses_action.append(train_loss['action'])
            train_losses.append(train_loss['full'])

            if self.scheduler is not None:
                self.scheduler.step()
            

            if self.world_size > 1:
                torch.distributed.barrier()
            
            if self.logger is not None and  i % self.log_freq == 0 and (self.world_size == 1 or self.rank==0):
                logs['time/training'] = time.time() - train_start
                logs['time/total'] = time.time() - self.start_time
                logs['optimizer/lr'] = self.optimizer.param_groups[0]['lr']
                global_step = iter_num * len(self.train_dataloader) + i
                self.logger.log_dict(logs, global_step)
                logs['training/train_loss_mean'] = np.mean(train_losses)
                logs['training/train_loss_std'] = np.std(train_losses)
                logs['training/train_loss_action_mean'] = np.mean(train_losses_action)
                logs['training/train_loss_action_std'] = np.std(train_losses_action)
            
            if self.save_dir is not None and i % self.model_save_freq == 0 and (self.world_size == 1 or self.rank==0):
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'last.pt'))
                self.saved_model_number += 1
            
            if self.save_dir is not None and i % 5000 == 0 and (self.world_size == 1 or self.rank==0):
                global_step = iter_num * len(self.train_dataloader) + i
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_step_{global_step}.pt'))

            if print_logs and i % self.log_freq == 0 and (self.world_size == 1 or self.rank==0):
                for k in self.diagnostics:
                    logs[k] = self.diagnostics[k]
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
        return logs

    def train_step(self, batch):

        proprio, depth, actions, timesteps, attention_mask = batch 

        action_target = torch.clone(actions)
        
        if self.add_proprio_noise:
            noise = torch.zeros_like(proprio)
            noise[...,:7] = torch.randn_like(proprio[...,:7])*self.noise_arm
            noise[...,7:] = torch.randn_like(proprio[...,7:])*self.noise_hand
            proprio = proprio + noise


        if self.world_size > 1:
            action_preds, _ = self.ddp_model.forward(
                        proprio, depth, timesteps=timesteps, attention_mask=attention_mask,)
        
        else:
            action_preds, _ = self.model.forward(
                        proprio, depth, timesteps=timesteps, attention_mask=attention_mask,)
        

        act_dim = action_preds.shape[2]

        if attention_mask is not None:
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]


        loss_action = self.loss_fn(action_preds, action_target)
        loss = loss_action


        self.optimizer.zero_grad()
        loss.backward()
        if self.world_size > 1:
            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), .25)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)

        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss_action.detach().cpu().item()
        
        return_dict = {'action': loss_action.detach().cpu().item(),
                        'full': loss.detach().cpu().item()
                        }
        
        return return_dict 
