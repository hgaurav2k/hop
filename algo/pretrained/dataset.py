import torch
from torch.utils.data import Dataset, DataLoader
import os 
import pickle as pkl 
from termcolor import cprint
class TrajectoryDataset(Dataset):

    def __init__(self, root,ctx_length=64,device='cuda'):
        """
        Args:
            data (Any): Your dataset (e.g., images, files, tensors).
            targets (Any): The labels or targets associated with your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(TrajectoryDataset, self).__init__()
        self.root = root 
        self.device = device 
        #assuming not many files in the directory 
        self.episodes = [pkl.load(open(os.path.join(root,episode),'rb')) for episode in os.listdir(root)]
        self.ctx = ctx_length
        self.ep_lens = torch.tensor([(len(episode)- self.ctx+1) for episode in self.episodes])
        self.cumsum = torch.cumsum(self.ep_lens,0)
        self.visualise()

    def visualise(self):
        """
        Visualise the dataset.
        """
        cprint(f"Number of episodes: {len(self.episodes)}",color='green',attrs=['bold'])
        cprint(f"Number of examples: {torch.sum(self.ep_lens)}",color='green',attrs=['bold'])
        cprint(f"Proprio dimension: {len(self.episodes[0]['robot_state'][0])}",color='green',attrs=['bold'])
        cprint(f"Action dimension: {len(self.episodes[0]['action'][0])}",color='green',attrs=['bold'])

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
        ep = self.episodes[ep_idx]
        idx = index - torch.sum(self.ep_lens[:ep_idx])
        return {
            'state': torch.tensor(ep['robot_state'][idx:idx+self.ctx]).to(self.device),
            'action': torch.tensor(ep['action'][idx:idx+self.ctx]).to(self.device),
            'timesteps': torch.tensor(torch.arange(idx,idx+self.ctx)).to(self.device),
        }



def collate_fn(batch):

    state = torch.stack([torch.tensor(item['state']) for item in batch])
    action = torch.stack([torch.tensor(item['action']) for item in batch])
    timesteps = torch.stack([torch.tensor(item['timesteps']) for item in batch])
    attention_mask = None 

    return state, action, timesteps, attention_mask 



