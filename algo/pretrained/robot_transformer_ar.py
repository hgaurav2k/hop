import numpy as np
import torch
import torch.nn as nn
import random 
import transformers
from algo.models.running_mean_std import RunningMeanStd
from .transformer import GPT2Model
from isaacgym import gymtorch
from collections import deque
from termcolor import cprint

np.set_printoptions(precision=3)


class RobotTransformerAR(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            max_ep_len=4096,
            cfg=None
    ):
        
        super(RobotTransformerAR, self).__init__()

        self.proprio_dim = cfg.pretrain.model.proprio_dim
        self.act_dim = cfg.pretrain.model.action_dim 
        self.pc_num = cfg.pretrain.model.pc_num
        self.max_ep_len = max_ep_len
        self.device = cfg.pretrain.device
        self.time_shift = cfg.pretrain.training.time_shift
        self.modality_aligned = cfg.pretrain.training.modality_aligned
        self.history_fill = self.time_shift + 1
        self.action_tanh = cfg.pretrain.model.action_tanh 
        self.action_input = cfg.pretrain.model.action_input
        if cfg.pretrain.model.action_input:
            self.n_ctx = 3*cfg.pretrain.model.context_length
        else:
            self.n_ctx = 2*cfg.pretrain.model.context_length

        self.cfg = cfg

        self.hidden_size = cfg.pretrain.model.hidden_dim

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            hidden_size = cfg.pretrain.model.hidden_dim, 
            n_embd = cfg.pretrain.model.hidden_dim,
            n_head = cfg.pretrain.model.n_head, 
            n_layer = cfg.pretrain.model.n_layer,
            resid_pdrop=cfg.pretrain.model.resid_pdrop,
            embd_pdrop=cfg.pretrain.model.embd_pdrop,
            attn_pdrop=cfg.pretrain.model.attn_pdrop,
            n_ctx = self.n_ctx
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(self.n_ctx+1, self.hidden_size) #1 extra for padding
        # self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_proprio = torch.nn.Linear(self.proprio_dim, self.hidden_size)

        #right now, action is the target with position control 
        if self.action_input:
            self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_pc  = nn.Sequential(
            nn.Linear(3,self.hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.MaxPool2d((self.pc_num,1))
        ) #PointNet 
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if self.action_tanh else []))
        )
        self.predict_proprio = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.proprio_dim)])
        )
        if self.action_input:
            self.predict_pc = nn.Sequential(
                *([nn.Linear(self.hidden_size, 3*self.pc_num)])
            )
        # self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, proprio, object_pc, action=None, timesteps=None, attention_mask=None):

        batch_size, seq_length = proprio.shape[0], proprio.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(self.device)

        # embed each modality with a different head
        proprio_embeddings = self.embed_proprio(proprio)
        if self.action_input:
            action_embeddings = self.embed_action(action)
        pc_embeddings = self.embed_pc(object_pc).squeeze(-2)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        proprio_embeddings = proprio_embeddings + time_embeddings
        if self.action_input:
            action_embeddings = action_embeddings + time_embeddings
        pc_embeddings = pc_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        if self.action_input:
            stacked_inputs = torch.stack(
                (proprio_embeddings, pc_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        else:
            stacked_inputs = torch.stack(
                (proprio_embeddings, pc_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        #need to check this 
        if self.action_input:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length) 
        else:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # states (0), or actions (1); i.e. x[:,1,t] is the token for s_t
        if self.action_input:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        
        if self.action_input:
            if self.modality_aligned:
                action_preds = self.predict_action(x[:,2])  # predict next action given the current action
                next_proprio_preds = self.predict_proprio(x[:,0])
                pc_preds = self.predict_pc(x[:,1])
            else:
                action_preds = self.predict_action(x[:,1])  # predict action given pc and proprio
                next_proprio_preds = self.predict_proprio(x[:,2])
                pc_preds = self.predict_pc(x[:,0])
            # put predictions in a dict
            pred_dict = {
                'action': action_preds,
                'next_proprio': next_proprio_preds,
                'pc': pc_preds
            }
        else:
            action_preds = self.predict_action(x[:,1])
            next_proprio_preds = self.predict_proprio(x[:,0])
            # put predictions in a dict
            pred_dict = {
                'action': action_preds,
                'next_proprio': next_proprio_preds
            }

        return pred_dict, pc_embeddings 

    @torch.no_grad()
    def get_action(self, proprio, object_pc, actions, **kwargs):

        bs = proprio.shape[0]
        n_ctx = proprio.shape[1]

        timesteps = torch.arange((n_ctx)).unsqueeze(0).repeat(bs,1).to(self.device)

        if self.action_input:
            pred_dict, _ = self.forward(
                proprio, object_pc, actions,timesteps, attention_mask=None, **kwargs)
        else:
            pred_dict, _ = self.forward(
                proprio, object_pc, timesteps=timesteps, attention_mask=None, **kwargs)

        #can implement kv caching and stuff here or use a generic transformer implementation
        #shift to a generic faster HF transformer with trainer as next step

        action_preds = pred_dict['action'][:, -self.history_fill]

        return 
    
        
    @torch.no_grad()
    def run_multi_env(self,env, cfg=None, **kwargs):
        
        self.eval()

        next_obs_dict = env.reset()

        num_envs = env.num_envs

        episode_rewards = torch.zeros((num_envs,)).to(self.device)
        episode_lengths = torch.zeros((num_envs,)).to(self.device)

        info_dict = {'episode_reward': [], 'episode_length': []}

        q_limits = {'lower': env.arm_hand_dof_lower_limits,
                    'upper': env.arm_hand_dof_upper_limits}

        def scale_q(q, limits):
            q = (q - limits['lower'][None]) / (limits['upper'][None] - limits['lower'][None])
            q = 2 * q - 1
            return q

        # Unscale the actions
        def unscale_q(q, limits):
            q = 0.5 * (q + 1) * (limits['upper'][None] - limits['lower'][None]) + limits['lower'][None]
            return q

        use_residuals = kwargs.get('use_residuals', False)

        assert not use_residuals, "Residuals not supported in this mode"

        done = False

        # assert self.time_shift == 0, "Time shift not supported in this mode"
        if self.time_shift > 0:
            print("Time shift not fully supported in multiple envs. We'll still use it, but should be fixed.")
        timestep = 0

        while True:

            # New data coming from env
            action_hist_input = next_obs_dict['action_buf'].clone()
            if not self.cfg.pretrain.model.scale_action:
                action_hist_input = unscale_q(action_hist_input, q_limits)
            action_hist_input = torch.cat((action_hist_input, torch.zeros_like(action_hist_input[:,-1:])), dim=1)
            proprio_hist_input = next_obs_dict['proprio_buf'].clone()
            if self.cfg.pretrain.model.scale_proprio:
                proprio_hist_input = scale_q(proprio_hist_input, q_limits)

            obj_pc_hist_input = next_obs_dict['pc_buf'].clone()
            attn_mask = next_obs_dict['attn_mask'].clone()
            ts = next_obs_dict['timesteps'].clone().long()

            with torch.no_grad():
                if self.action_input:
                    pred_dict, _ = self.forward(
                        proprio_hist_input, obj_pc_hist_input, action_hist_input, ts, attention_mask=attn_mask, **kwargs)
                else:
                    pred_dict, _ = self.forward(
                        proprio_hist_input, obj_pc_hist_input, timesteps=ts, attention_mask=attn_mask, **kwargs)

                pred_action = pred_dict['action'][:, -1]
                
                if not self.cfg.pretrain.model.scale_action:
                    pred_action = scale_q(pred_action, q_limits)
            
            
            next_obs_dict, r, done, info = env.step(pred_action.clone())
            episode_rewards += r

            at_reset_env_ids = torch.where(done)[0]
            for env_id in at_reset_env_ids:
                info_dict['episode_reward'].append(episode_rewards[env_id].item())
                info_dict['episode_length'].append(episode_lengths[env_id].item())
                episode_rewards[env_id] = 0
                episode_lengths[env_id] = 0

            timestep += 1 
            episode_lengths += 1

            if timestep > 2*env.max_episode_length:
                break

        # Transfor the list into a tensor
        info_dict['episode_reward'] = torch.tensor(info_dict['episode_reward'])
        info_dict['episode_length'] = torch.tensor(info_dict['episode_length'])
        info_dict = env.linearize_info(info_dict)
        
        return info_dict