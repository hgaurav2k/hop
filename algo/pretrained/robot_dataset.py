import torch
from torch.utils.data import Dataset, DataLoader
import os 
import pickle as pkl 
from termcolor import cprint
import numpy as np
class RobotDataset(Dataset):

    def __init__(self, root=None, cfg=None):
        """
        Args:
            data (Any): Your dataset (e.g., images, files, tensors).
            targets (Any): The labels or targets associated with your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert root is not None, "Please provide the root directory of the dataset"
        assert os.path.exists(root), f"The directory {root} does not exist"
        super(RobotDataset, self).__init__()
        self.root = root 
        print(f"Loading dataset from {root}")
        self.device = cfg.pretrain.device
        self.ctx = cfg.pretrain.model.context_length
        self.scale_action = cfg.pretrain.model.scale_action
        self.scale_proprio = cfg.pretrain.model.scale_proprio
        # set variable to store the episodes
        self.episodes_npy = []
        self.ep_lens = []
        # self.dt = kwargs.get('dt', 0.008333) # 120 Hz
        # self.dt = np.float32(self.dt)
        self.use_residuals = cfg.pretrain.training.use_residuals
        # get all folders of depth 2 in the directory
        subjects_dir = [os.path.join(root,episode) for episode in os.listdir(root) if os.path.isdir(os.path.join(root,episode))]
        # get all subfolders of depth 2 in subjects_dir
        self.episodes_dir = [os.path.join(subject,episode) for subject in subjects_dir for episode in os.listdir(subject) if os.path.isdir(os.path.join(subject,episode))]
        self.episodes_dir = sorted(self.episodes_dir)

        assert len(self.episodes_dir) > 0, f"No episodes found in the directory {root}"
        # load all the episodes
        for episode in self.episodes_dir:
            self.load_episode_fnames(episode)

        assert len(self.episodes_npy) > 0, f"No trajectories found in the directory {root}"
        # save the min, max, and mean of the episode lengths
        self.min_ep_len = np.min(self.ep_lens)
        self.max_ep_len = np.max(self.ep_lens)
        self.mean_ep_len = np.mean(self.ep_lens)
        cprint(f"Min episode length: {self.min_ep_len}, Max episode length: {self.max_ep_len}, Mean episode length: {self.mean_ep_len}",color='cyan',attrs=['bold'])
        self.ep_lens = torch.tensor(self.ep_lens)
        self.cumsum = torch.cumsum(self.ep_lens,0)
        self.visualise()

        # IG lower and upper limits
        self.limits = {'upper': [6.2832, 2.0944, 6.2832, 3.9270, 6.2832, 3.1416, 6.2832, 0.4700, 1.6100, 1.7090, 1.6180, 1.3960,
                                  1.1630, 1.6440, 1.7190, 0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180],
                       'lower': [-6.2832, -2.0590, -6.2832, -0.1920, -6.2832, -1.6930, -6.2832, -0.4700, -0.1960, -0.1740, -0.2270,
                                  0.2630, -0.1050, -0.1890, -0.1620, -0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270]}

        
        self.limits['upper'] = np.array(self.limits['upper']).astype(np.float32)
        self.limits['lower'] = np.array(self.limits['lower']).astype(np.float32)    


    def load_episode_fnames(self, episode_dir:str):
        """
        Load the episodes filenames.
        """
        for episode_fname in sorted(os.listdir(episode_dir)):
            # continue if the file is not a npy file
            if not episode_fname.endswith('.npy'):
                continue
            ep = np.load(os.path.join(episode_dir,episode_fname), allow_pickle=True).item()
            self.episodes_npy.append(ep)
            # load the file and get the length
            eplen = len(ep['robot_qpos']) - self.ctx + 1

            assert eplen > 0, f"Episode length is less than the context length {self.ctx}"
            
            self.ep_lens.append(eplen)

    def scale_q(self, q):
        """
        Scale the proprioceptive data to be between -1 and 1.
        """
        q = (q - self.limits['lower']) / (self.limits['upper'] - self.limits['lower'])
        q = 2 * q - 1
        return q
    
    def change_order(self, q):
        IG_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 11, 12, 13, 14, 15, 16, 17, 18]
        return q[:,IG_mapping]

    def visualise(self):
        """
        Visualise the dataset.
        """
        cprint(f"Number of episodes: {len(self.episodes_npy)}",color='green',attrs=['bold'])
        cprint(f"Number of examples: {torch.sum(self.ep_lens)}",color='green',attrs=['bold'])
        # Load the first episode to get the dimension of the proprio and action
        ep = self.episodes_npy[0]
        cprint(f"Proprio dimension: {len(ep['robot_qpos'][0])}",color='green',attrs=['bold'])
        cprint(f"Action dimension: {len(ep['target_qpos'][0])}",color='green',attrs=['bold'])

    def __len__(self):
        """Returns the size of the dataset."""
        return torch.sum(self.ep_lens).item()

    def __getitem__(self, index):
        """
        Generates one sample of data.
        
        Args:
            index (int): The index of the item in the dataset
        
        Returns:
            sample (Any): The data sample corresponding to the given index.
            target (Any): The target corresponding to the given data sample.
        """

        ep_idx = torch.searchsorted(self.cumsum, index, right=True) 
        # open the pickle file
        idx = index - torch.sum(self.ep_lens[:ep_idx])
        ep = self.episodes_npy[ep_idx]
        action_npy = np.stack(ep['target_qpos'][idx:idx+self.ctx])
        proprio_npy = np.stack(ep['robot_qpos'][idx:idx+self.ctx])
        # Put in IG order
        action = self.change_order(action_npy)
        proprio = self.change_order(proprio_npy)
        # Scale the proprioceptive data in [-1,1]
        # For the first 7 elements of the action vector, predict the residual with respect to the previous action
        if self.use_residuals:
            action_res = np.concatenate([np.zeros((1,action.shape[1])), np.diff(action, axis=0)], axis=0)
            action_res[0] = action[0] - proprio[0]
            action_res = action_res.astype(np.float32)
            action = action_res / self.dt

        if self.scale_proprio:
            proprio = self.scale_q(proprio)
        if self.scale_action:
            action = self.scale_q(action)
        
        obj_pc = np.stack(ep['object_pc'][idx:idx+self.ctx])

        return {
            'proprio': proprio,
            'action': action,
            'obj_pc': obj_pc,
            'timesteps': np.arange(self.ctx),
        }


def collate_fn(batch):

    proprio = np.stack([item['proprio'] for item in batch])
    object_pc = np.stack([item['obj_pc'] for item in batch])
    action = np.stack([item['action'] for item in batch])
    timesteps = np.stack([item['timesteps'] for item in batch])
    attention_mask = None 

    proprio = torch.tensor(proprio, dtype=torch.float32, requires_grad=False)
    object_pc = torch.tensor(object_pc, dtype=torch.float32, requires_grad=False)
    action = torch.tensor(action, dtype=torch.float32, requires_grad=False)
    timesteps = torch.tensor(timesteps, dtype=torch.long, requires_grad=False)

    return proprio, object_pc, action, timesteps, attention_mask 
