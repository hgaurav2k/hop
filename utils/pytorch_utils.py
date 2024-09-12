import torch 
import numpy as np 
from copy import deepcopy

device=None 


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def to_torch(element,device):

    if isinstance(element,dict):
        
        new_element = deepcopy(element)
        for key in element:
            new_element[key] = to_torch(element[key],device)
        return new_element 
    
    elif isinstance(element,list):
        try: 
            return torch.tensor(element).float().to(device)
        except:
            return element 
    
    elif isinstance(element,np.ndarray):
        return torch.from_numpy(element).float().to(device)
    
    else:
        return element 

