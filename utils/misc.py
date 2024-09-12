# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import os
import torch
import shlex
import random
import subprocess
import numpy as np
import torch.distributed as dist
from termcolor import cprint 
import matplotlib.pyplot as plt
import json 
import utils.pytorch_utils as  ptu

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def git_diff_config(name):
    cmd = f'git diff --unified=0 {name}'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def get_world_size():
    # return number of gpus
    if 'LOCAL_WORLD_SIZE' in os.environ.keys():
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        world_size = 1
    return world_size


def multi_gpu_aggregate_stats(values):
    if type(values) is not list:
        single_item = True
        values = [values]
    else:
        single_item = False
    rst = []
    for v in values:
        if type(v) is list:
            v = torch.stack(v)
        if get_world_size() > 1:
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v = v / get_world_size()
        if v.numel() == 1:
            v = v.item()
        rst.append(v)
    if single_item:
        rst = rst[0]
    return rst


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed

def get_rank():
    return int(os.getenv('LOCAL_RANK', '0'))


def vulkan_device_id_from_cuda_device_id(orig: int) -> int:
        """Map a CUDA device index to a Vulkan one.

        Used to populate the value of `graphic_device_id`, which in IsaacGym is a vulkan
        device ID.

        This prevents a common segfault we get when the Vulkan ID, which is by default 0,
        points to a device that isn't present in CUDA_VISIBLE_DEVICES.
        """
        # Get UUID of the torch device.
        # All of the private methods can be dropped once this PR lands:
        #     https://github.com/pytorch/pytorch/pull/99967
        try:
            cuda_uuid = torch.cuda._raw_device_uuid_nvml()[
                torch.cuda._parse_visible_devices()[orig]
            ]  # type: ignore
            assert cuda_uuid.startswith("GPU-")
            cuda_uuid = cuda_uuid[4:]
        except AttributeError:
            cprint('detect cuda / vulkan relation can only be done for pytorch 2.0', 'red', attrs=['bold'])
            return get_rank()

        try:
            vulkaninfo_lines = subprocess.run(
                ["vulkaninfo"],
                # We unset DISPLAY to avoid this error:
                # https://github.com/KhronosGroup/Vulkan-Tools/issues/370
                env={k: v for k, v in os.environ.items() if k != "DISPLAY"},
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            ).stdout.split("\n")
        except FileNotFoundError:
            cprint(
                "vulkaninfo was not found; try `apt install vulkan-tools` or `apt install vulkan-utils`."
            , 'red', attrs=['bold'])
            return get_rank()

        vulkan_uuids = [
            s.partition("=")[2].strip()
            for s in vulkaninfo_lines
            if s.strip().startswith("deviceUUID")
        ]
        vulkan_uuids = list(dict(zip(vulkan_uuids, vulkan_uuids)).keys())
        vulkan_uuids = [uuid for uuid in vulkan_uuids if not uuid.startswith("0000")]
        out = vulkan_uuids.index(cuda_uuid)
        print(f"Using graphics_device_id={out}", cuda_uuid)
        return out


def compile_per_asset(env,reward_arr,success_arr, name, title=""):
    assert env.num_envs == reward_arr.shape[0]

    #make a histogram using matplotlib
    
    unique_objects = list(set(env.object_names))
    unique_reward_arr = np.zeros(len(unique_objects))
    unique_success_arr = np.zeros(len(unique_objects))

    mean_success = np.mean(success_arr)
    mean_reward = np.mean(reward_arr)

    cnt = np.zeros(len(unique_objects))
    for k,v,w in zip(env.object_names,reward_arr,success_arr):
        idx = unique_objects.index(k)
        unique_reward_arr[idx] += v
        unique_success_arr[idx] += w
        cnt[idx] += 1
    

    unique_reward_arr /= cnt
    unique_success_arr /= cnt

    data = {k:(v,w) for k,v,w in zip(unique_objects,unique_reward_arr,unique_success_arr)}  


    json.dump(data,open(f'reward_analysis_{name}.json','w'))
    plt.rcParams['figure.figsize'] = [50,50]
    idxs = unique_reward_arr.argsort()

    plt.bar([unique_objects[j] for j in list(idxs)], unique_reward_arr[idxs])

    #mean line at mean_reward 
    plt.axhline(y=mean_reward, color='r', linestyle='-',label=f'Mean Reward {mean_reward}')
    #write x coordinates on the bar itself with bold font
    plt.rcParams.update({'font.size': 30})  # You can adjust the size as needed
    for i, v in enumerate(list(unique_reward_arr[idxs])):
        plt.text(i-.25, v + 0.01, unique_objects[i],rotation=90,fontweight='bold')
    # plt.xticks(list(range(len(env.object_names[idxs]))), env.object_names[idxs], rotation=90, fontweight='bold')

    #increase figure size 
    # Adding titles and labels
    plt.xlabel('Objects')
    plt.ylabel('Values')
    plt.title(f'Object Names v/s Returns {title}')
    plt.xticks(rotation=90)  # Rotate labels to 45 degrees
    # Optional: Change the font size for better readability
    # Show plot

    
    plt.savefig(f'reward_analysis_{name}.png')
    plt.close()

    # mass_list = ptu.to_numpy(env.priv_buf[:,4])
    # # data = {k:v for k,v in zip(mass_list,reward_arr)}
    # # json.dump(data,open('reward_analysis.json','w'))
    # plt.plot(mass_list,unique_reward_arr,'o')
    # plt.xlabel('Mass')
    # plt.ylabel('Rewards')
    # plt.savefig('mass_vs_reward.png')
    # plt.close()


    if unique_success_arr is not None:
        # data = {k:v for k,v in zip(mass_list,reward_arr)}
        # json.dump(data,open('reward_analysis.json','w'))
        # plt.plot(mass_list,unique_success_arr,'o')
        # plt.xlabel('Mass')
        # plt.ylabel('Rewards')
        # plt.savefig('mass_vs_success.png')
        # plt.close()

        plt.bar([unique_objects[j] for j in list(idxs)], unique_success_arr[idxs])
        #mean success 
        plt.axhline(y=mean_success, color='r', linestyle='-',label=f'Mean Success {mean_success}')
        
        # plt.rcParams.update({'font.size': 100})  # You can adjust the size as needed
        # #write x coordinates on the bar itself with bold font
        # for i, v in enumerate(list(unique_success_arr[idxs])):
        #     plt.text(i-.25, v + 0.01, unique_objects[i],rotation=90,fontweight='bold')
        
        # plt.xticks(list(range(len(env.object_names[idxs]))), env.object_names[idxs], rotation=90, fontweight='bold')
        plt.xlabel('Objects')
        plt.ylabel('Values')
        plt.title(f'Object Names v/s Returns {title}')
        # for i, v in enumerate(list(unique_reward_arr[idxs])):
        #     plt.text(i-.25, v + 0.01, unique_objects[i],rotation=90,fontweight='bold')
        plt.xticks(rotation=90)  # Rotate labels to 45 degrees
        # Optional: Change the font size for better readability
        # Show plot
        plt.savefig(f'success_analysis_{name}.png')
        plt.close()

def depth_img_resize(depth_img,size):

    if depth_img.ndim == 4:
        depth_img = torch.nn.functional.interpolate(depth_img, size=(size, size), mode='bilinear', align_corners=False)
    elif depth_img.ndim == 3:
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
        depth_img = depth_img.squeeze(0)
    else:
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0).unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
        depth_img = depth_img.squeeze(0).squeeze(0)

    return depth_img


def depth_img_center_crop(depth_img, size=256):
    #center crop the image

    if depth_img.ndim == 3:

        _,h, w = depth_img.shape
        h_start = (h - size) // 2
        w_start = (w - size) // 2
        depth_img = depth_img[:, h_start:h_start+size, w_start:w_start+size]
    
    elif depth_img.ndim == 4:
        _,_,h, w = depth_img.shape
        h_start = (h - size) // 2
        w_start = (w - size) // 2
        depth_img = depth_img[:,:,h_start:h_start+size, w_start:w_start+size]

    else:
        h, w = depth_img.shape
        h_start = (h - size) // 2
        w_start = (w - size) // 2
        depth_img = depth_img[h_start:h_start+size, w_start:w_start+size]
        
    return depth_img


def batch_rotate_images_torch(I, R):
    N, C, H, W = I.shape
    device = I.device  # Ensure operations are on the same device as the input tensor
    R_inv = torch.inverse(R)  # Invert each matrix in the batch

    # Create grid of coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    x = x - W // 2
    y = y - H // 2

    # Flatten x and y for easier broadcasting and replicate for each image in the batch
    x_flat = x.flatten().repeat(N, 1)
    y_flat = y.flatten().repeat(N, 1)
    
    # Create homogeneous coordinates
    ones = torch.ones_like(x_flat)
    homogeneous_coords = torch.stack([x_flat, y_flat, ones], dim=-1).float()

    # Apply the inverse rotation matrices
    rotated_coords = torch.bmm(homogeneous_coords, R_inv.transpose(1, 2))
    
    # Reshape rotated coordinates back to image dimensions and adjust centers
    x_rot = rotated_coords[:, :, 0].view(N, H, W) + W // 2
    y_rot = rotated_coords[:, :, 1].view(N, H, W) + H // 2

    # Clip coordinates to image dimensions
    x_rot = torch.clamp(x_rot, 0, W - 1).long()
    y_rot = torch.clamp(y_rot, 0, H - 1).long()

    # Gather pixels from the original images using advanced indexing
    batch_indices = torch.arange(N, device=device)[:, None, None]
    channel_indices = torch.arange(C, device=device)[:, None]


    I_prime = I[batch_indices, channel_indices, y_rot, x_rot]


    return I_prime

def euler_to_rotation_matrix_torch(euler_angles):
    """
    Convert an array of Euler angles (XYZ) to rotation matrices using PyTorch, without loops.
    
    :param euler_angles: An Nx3 tensor of Euler angles
    :returns: An Nx3x3 tensor of corresponding rotation matrices
    """
    # Extract individual angles
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]

    # Precompute sines and cosines of the angles
    sx, cx = torch.sin(roll), torch.cos(roll)
    sy, cy = torch.sin(pitch), torch.cos(pitch)
    sz, cz = torch.sin(yaw), torch.cos(yaw)

    # Compute the elements of the rotation matrix
    r00 = cy * cz
    r01 = -cy * sz
    r02 = sy
    r10 = cx * sz + cz * sx * sy
    r11 = cx * cz - sx * sy * sz
    r12 = -cy * sx
    r20 = sx * sz - cx * cz * sy
    r21 = cz * sx + cx * sy * sz
    r22 = cx * cy

    # Stack the rotation matrices in the shape (N, 3, 3)
    rotation_matrices = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    return rotation_matrices


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean

    