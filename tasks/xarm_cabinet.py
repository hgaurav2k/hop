# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import math
import os
import random
import tempfile
from copy import copy
from typing import Dict, Any, Tuple, List, Set
import utils.pytorch_utils as ptu
from isaacgym import gymapi, gymtorch, gymutil
from torch import Tensor
import torchvision
from utils.hand_arm_utils import DofParameters, populate_dof_properties
from tasks.base.vec_task import VecTask
from tasks.torch_jit_utils import *
from scipy.spatial.transform import Rotation as R
from utils.hand_arm_utils import tolerance_curriculum, tolerance_successes_objective
from utils.urdf_utils import read_xml, load_asset_files_public, load_asset_files_ycb, load_asset_files_ycb_lowmem, get_link_meshes_from_urdf
import trimesh 
import matplotlib.pyplot as plt
from termcolor import cprint 
from utils.randomization_utils import randomize_table_z, randomize_friction, randomize_object_scale, randomize_object_mass
from utils.misc import depth_img_center_crop, depth_img_resize, batch_rotate_images_torch, euler_to_rotation_matrix_torch
import json
import torchvision.transforms as T

class AllegroXarmCabinet(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture , force_render, pretrain_cfg=None):

        self.cfg = cfg

        self.frame_since_restart: int = 0  # number of control steps since last restart across all actors

        # self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["Xarm7"]
        if self.cfg["env"]["use_allegro"]:
            self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["Xarm7_allegro"]
        elif self.cfg["env"]["use_leap"]:
            self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["Xarm7_leap_hand"]
        else:
            raise Exception("Unknown hand type!")

        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]

        self.privileged_actions = self.cfg["env"]["privilegedActions"]
        self.privileged_actions_torque = self.cfg["env"]["privilegedActionsTorque"]
        self.log_video = False 

        # 4 joints for index, middle, ring, and thumb and 7 for ka arm
        self.num_arm_dofs = 7
        self.num_finger_dofs = 4
        self.num_allegro_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_allegro_fingertips
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        self.num_allegro_kuka_actions = self.num_hand_arm_dofs
        if self.privileged_actions:
            self.num_allegro_kuka_actions += 3
        
        self.point_cloud_sampled_dim = 100 #hard-coding for now
        self.hand_joint_point_cloud_sampled_dim = 100 #hard-coding for now 

        self.dof_params: DofParameters = DofParameters.from_cfg(self.cfg)

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"]["resetDofPosRandomIntervalFingers"]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.num_props = 0


        self.hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.old_action_space = self.cfg["env"]["useOldActionSpace"]
        self.limit_arm_delta_target = self.cfg["env"]["limitArmDeltaTarget"]
        if pretrain_cfg is not None:
            self.use_residuals = pretrain_cfg["training"]["use_residuals"]
            self.pretrain_dt = pretrain_cfg["training"]["dt"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.enable_camera_sensors = self.cfg["env"]["enableCameraSensors"] 
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]
        
        self._setup_visual_observation(self.cfg['env']['rgbd_camera'])


        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.thumb_dist_reward_scale = self.cfg["env"]["thumbDistRewardScale"]
        self.handle_dist_reward_scale = self.cfg["env"]["handleDistRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_bonus_reward_scale = self.cfg["env"]["openBonusRewardScale"]
        self.goal_dist_reward_scale = self.cfg["env"]["goalDistRewardScale"]
        self.open_pose_reward_scale = self.cfg["env"]["openPoseRewardScale"]
        self.goal_bonus_reward_scale = self.cfg["env"]["goalBonusRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.enable_proprio_history = self.cfg['env']['enableProprioHistory']
        self.use_obs_as_prop = self.cfg["env"]["useObsAsProp"]
        self.enable_action_history = self.cfg['env']['enableActionHistory']
        self.enable_point_cloud = self.cfg["env"]["enablePointCloud"]
        self.enable_attn_mask = self.cfg["env"]["enableAttnMask"]
        self.input_priv = self.cfg["env"]["input_priv"]
        
        self.keypoint_scale = self.cfg["env"]["keypointScale"]
        self.point_cloud_scale = self.cfg["env"]["pointCloudScale"]

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (self.control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time),
            print("New episode length: ", self.max_episode_length)

        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")

        self.cabinet_asset_file = "sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"
        self.ball_asset_file = "urdf/ball.urdf"
        self.table_asset_file = "urdf/slider2.urdf"
        
        self.keypoints_offsets = self._object_keypoint_offsets()

        self.num_keypoints = len(self.keypoints_offsets)

        if self.cfg["env"]["use_allegro"]:

            # self.allegro_fingertips = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
            self.allegro_fingertips = ["link_3.0", "link_7.0", "link_11.0", "link_15.0"]

            self.hand_joints = self.allegro_fingertips  #for now

            self.fingertip_offsets = np.array(
                [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]], dtype=np.float32
            )
            self.palm_offset = np.array([-0.00, -0.02, 0.16], dtype=np.float32)


        elif self.cfg["env"]["use_leap"]:

            self.allegro_fingertips = self.cfg["env"]["leapFingers"]

            self.hand_joints = self.allegro_fingertips #for now

            if self.cfg["env"]["useDIPFinger"]:
                self.allegro_fingertips = self.cfg["env"]["leapDIP"]

            self.fingertip_offsets = 0.*np.array(
                [[0.0, -0.05, -0.0], [0, -0.05, 0.0], [0, -0.05, 0], [0.0, -0.06, 0.0]], dtype=np.float32
            )
            self.palm_offset = np.array([0., 0.05, 0.1], dtype=np.float32)

        assert self.num_allegro_fingertips == len(self.allegro_fingertips)


        self.full_state_size = 77 

        num_states = self.full_state_size

        self.up_axis = "z"

        self.cfg["env"]["numObservations"] = 77 
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.num_allegro_kuka_actions

        self.cfg["device_type"] = sim_device.split(":")[0]
        self.cfg["device_id"] = int(sim_device.split(":")[1])
        self.cfg["headless"] = headless


        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        if self.viewer is not None:
            #cam_pos for simulation camera
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # volume to sample target position from
        # if self.use_fixed_height_as_target:
        #     self.goal_height = self.cfg["env"]["goalHeight"]
        # else:
        #     target_volume_origin = np.array([0, 0.05, 0.8], dtype=np.float32)
        #     target_volume_extent = np.array([[-0.4, 0.4], [-0.05, 0.3], [-0., 0.25]], dtype=np.float32)
        #     self.target_volume_origin = torch.from_numpy(target_volume_origin).to(self.device).float()
        #     self.target_volume_extent = torch.from_numpy(target_volume_extent).to(self.device).float()


        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_body_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)


        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.kuka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :23] #23 is the number of dofs in the allegro xarm
        self.kuka_dof_pos = self.kuka_dof_state[..., 0]
        self.kuka_dof_vel = self.kuka_dof_state[..., 1]

        # Cabinet dof state
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, 23:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]

        # (N, num_bodies, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        # (N, 2 + num_props, 13)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # (N, num_bodies, 3)
        self.contact_forces = gymtorch.wrap_tensor(rigid_body_force_tensor).view(self.num_envs, -1, 3)

        # Palm pos
        self.palm_pos = self.rigid_body_states[:, self.rigid_body_palm_ind, 0:3]

        # Finger pos
        self.index_pos = self.rigid_body_states[:, self.rigid_body_index_ind, 0:3]
        self.middle_pos = self.rigid_body_states[:, self.rigid_body_middle_ind, 0:3]
        self.ring_pos = self.rigid_body_states[:, self.rigid_body_ring_ind, 0:3]
        self.thumb_pos = self.rigid_body_states[:, self.rigid_body_thumb_ind, 0:3]

        # Finger rot
        self.index_rot = self.rigid_body_states[:, self.rigid_body_index_ind, 3:7]
        self.middle_rot = self.rigid_body_states[:, self.rigid_body_middle_ind, 3:7]
        self.ring_rot = self.rigid_body_states[:, self.rigid_body_ring_ind, 3:7]
        self.thumb_rot = self.rigid_body_states[:, self.rigid_body_thumb_ind, 3:7]

        # Drawer pose
        self.drawer_handle_pos = self.rigid_body_states[:, self.rigid_body_drawer_top_ind, 0:3]
        self.drawer_handle_rot = self.rigid_body_states[:, self.rigid_body_drawer_top_ind, 3:7]

        # Dof targets
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # Global inds
        self.global_indices = torch.arange(
            self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # Kuka dof pos and vel scaled
        self.kuka_dof_pos_scaled = torch.zeros_like(self.kuka_dof_pos)
        self.kuka_dof_vel_scaled = torch.zeros_like(self.kuka_dof_vel)

        # Finger to handle vecs
        self.index_to_handle = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.middle_to_handle = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ring_to_handle = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.thumb_to_handle = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Drawer open distance to goal
        self.to_goal = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)

        # Success counts
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.extras["successes"] = self.successes


        self.hand_arm_default_dof_pos = torch.zeros(self.num_hand_arm_dofs, dtype=torch.float, device=self.device)

        
        kuka_init_poses = {
            "v1": [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358],
            "v2": [-2.135, 0.843, 1.786, -0.903, -2.262, 1.301, -2.791],
            "v3": [1.54, -0.90, 0.89, 2.01, 1.42, -0.15, 1.00],
            "v4": [1.22, -0.45, -1.42, 2.92, 6.28, 0.08, -1.48],
            "v5": [4.06,-0.62,0.32,1.11,-0.32,-0.94,2.64],
            "v6": [-4.06,0.62,-0.32,-1.11,0.32,0.94,-2.64],
            "v7": [4.06,-0.62,0.32,-1.11,-0.32,-0.94,0.71],
            "v8": [0,-0.5,3.43,2.66,2.72,0,-0.36],
            "v9": [1.5458e-03, -5.0079e-01,  3.4346e+00,  2.6556e+00,  2.7197e+00, 1.5933e-02, -3.3825e-01],
            "v10": [1.5458e-03, -5.0079e-01,  3.4346e+00,  2.6556e+00,  2.7197e+00, 1.5933e-02, -3.3825e-01, 2.1337e-01, 1.7948e-01, 3.3804e-01, 4.3598e-02, 9.2495e-01, 2.0406e-01, 1.2707e+00, 2.4769e-01, 1.1064e-01, 7.5794e-01, 1.6942e-01, 1.2225e-01, 3.9004e-01, 5.3227e-01, 7.0137e-01, 1.2435e-01],
            "v11": [1.74, -0.39, -2.45, 3.03, -4.45, -0.76, 2.58, 2.1337e-01, 1.7948e-01, 3.3804e-01, 4.3598e-02, 9.2495e-01, 2.0406e-01, 1.2707e+00, 2.4769e-01, 1.1064e-01, 7.5794e-01, 1.6942e-01, 1.2225e-01, 3.9004e-01, 5.3227e-01, 7.0137e-01, 1.2435e-01],
            "v12": [1.22, -0.45, -1.42, 2.92, 6.28, 0.08, -1.48, 2.1337e-01, 1.7948e-01, 3.3804e-01, 4.3598e-02, 9.2495e-01, 2.0406e-01, 1.2707e+00, 2.4769e-01, 1.1064e-01, 7.5794e-01, 1.6942e-01, 1.2225e-01, 3.9004e-01, 5.3227e-01, 7.0137e-01, 1.2435e-01],
            "v13": [1.980484, -0.266612,  2.371495,  2.367739, -1.296472,  0.083561,  2.452218, -0.031805,  0.782409,  0.669216,  0.086214,  1.041541,  0.806434,  0.028384,  0.201165, -0.040333,  0.810362,  1.099423,  0.054407,  0.186913,  1.488713,  0.506201,  0.048111],
            "v14": [1.76, 0.30, 2.43, 2.26, -2.76, -1.45, 3.61, 0.04, 1.58, 0.31, 0.00, 1.02, 0.04, 0.84, 0.00, 0.18, 1.58, 0.28, 0.00, 0.31, 1.53, 0.35, 0.00],
            "v15": [-1.35, -0.30, 0.75, 1.0, 2.9, 1.07, -1.93], #[-1.35, -0.30, 0.75, 0.50  , 2.9, 1.07, 0.56]
            "v16": [-0.79, -0.30, 0.16, 0.43, 2.20, 0.72, -1.67]
        }
        #[-1.35, -0.30, 0.75, 0.5, 2.9, 1.07, -1.93]
        # [-1.35, -0.30, 0.75, 0.50  , 2.9, 1.07, 0.56]
        if self.cfg["env"]["initPoseVersion"] == "v10" or self.cfg["env"]["initPoseVersion"] == "v11" or self.cfg["env"]["initPoseVersion"] == "v12" or self.cfg["env"]["initPoseVersion"] == "v13" or self.cfg["env"]["initPoseVersion"] == "v14":
            self.hand_arm_default_dof_pos = torch.tensor(kuka_init_poses[self.cfg["env"]["initPoseVersion"]], dtype=torch.float, device=self.device)
        else:
            self.hand_arm_default_dof_pos[:7] = torch.tensor(kuka_init_poses[self.cfg["env"]["initPoseVersion"]], dtype=torch.float, device=self.device)



        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_hand_arm_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.rigid_body_force_tensor = gymtorch.wrap_tensor(rigid_body_force_tensor).view(self.num_envs, -1, 3)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.set_actor_root_state_object_indices: List[Tensor] = []

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        print("Num dofs: ", self.num_dofs)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)

        # true objective value for the whole episode, plus saving values for the previous episode
        # self.true_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
    
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.action_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )

        # how many steps we were within the goal tolerance
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.closest_fingertip_dist = -torch.ones(
            [self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device
        )

        # self.closest_joint_tgt_rel_pose_dist = -torch.ones(
        #     [self.num_envs, len(self.hand_joints)], dtype=torch.float, device=self.device
        # )

        self.closest_joint_tgt_rel_pose_dist = -torch.ones(
            [self.num_envs,len(self.hand_joints)], dtype=torch.float, device=self.device
        )

        self.closest_fingertip_shape_dist = -torch.ones([self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device)

        self.closest_palm_dist = -torch.ones(
            [self.num_envs, 1], dtype=torch.float, device=self.device
        )
        self.furthest_hand_dist = -torch.ones([self.num_envs], dtype=torch.float, device=self.device)

        self.finger_rew_coeffs = torch.ones(
            [self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device
        )


    
        # self.hand_forces = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # self.contact_forces = torch.zeros((self.num_envs, self.num_hand_arm_dofs), dtype=torch.float, device=self.device)

        reward_keys = [
            "raw_ih_dist_reward",
            "raw_mh_dist_reward",
            "raw_rh_dist_reward",
            "raw_th_dist_reward",
            "raw_around_handle_reward",
            "raw_open_bonus_reward",
            "raw_goal_reward",
            "raw_open_pose_reward",
            "raw_goal_bonus_reward",
            "raw_action_penalty",
            "ih_dist_reward",
            "mh_dist_reward",
            "th_dist_reward",
            "rh_dist_reward",
            "around_handle_reward",
            "open_bonus_reward",
            "goal_reward",
            "open_pose_reward",
            "goal_bonus_reward",
            "action_penalty"
        ]

        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys
        }

        self.last_curriculum_update = 0

        self.episode_root_state_tensors = [[] for _ in range(self.num_envs)]
        self.episode_dof_states = [[] for _ in range(self.num_envs)]

        self.eval_stats: bool = self.cfg["env"]["evalStats"]
        if self.eval_stats:
            self.last_success_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.success_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.total_num_resets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.successes_count = torch.zeros(
                self.max_consecutive_successes + 1, dtype=torch.float, device=self.device
            )
            from tensorboardX import SummaryWriter

            self.eval_summary_dir = "./eval_summaries"
            # remove the old directory if it exists
            if os.path.exists(self.eval_summary_dir):
                import shutil

                shutil.rmtree(self.eval_summary_dir)
            self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)


    def _setup_visual_observation(self, vo_config):
        self.rgbd_camera_config = vo_config
        self.enable_depth_camera = vo_config['enable_depth']
        self.rgbd_camera_width = self.cam_w = self.rgbd_camera_config['camera_width']
        self.rgbd_camera_height = self.cam_h = self.rgbd_camera_config['camera_height']
        self.rgbd_buffer_width = self.rgbd_camera_config['buffer_width']
        self.rgbd_buffer_height = self.rgbd_camera_config['buffer_height']
        self.num_cameras = self.rgbd_camera_config['num_cameras']
        self.cam_pose_rnd = self.rgbd_camera_config['randomize_camera_pose']
        self.cam_rot_rnd = self.rgbd_camera_config['randomize_camera_rot']
        # self.cam_pos = []
        # self.cam_target = []
        # for i in range(self.num_cameras):
        #     self.cam_pos.append(gymapi.Vec3(*self.rgbd_camera_config[f'cam{i}']['pos']))
        #     self.cam_target.append(gymapi.Vec3(*self.rgbd_camera_config[f'cam{i}']['target']))
        
        self.enable_wrist_camera = self.rgbd_camera_config['wrist_camera']
        self.camera_config = json.load(open(self.rgbd_camera_config['intrinsics']))
        # self.rgbd_center_crop = torchvision.transforms.CenterCrop((self.rgbd_buffer_height, self.rgbd_buffer_width))


    def scale_q(self,q):
        
        limits = {'upper': torch.tensor([6.2832, 2.0944, 6.2832, 3.9270, 6.2832, 3.1416, 6.2832, 0.4700, 1.6100, 1.7090, 1.6180, 1.3960,
                                  1.1630, 1.6440, 1.7190, 0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180]).to(self.device),
                'lower': torch.tensor([-6.2832, -2.0590, -6.2832, -0.1920, -6.2832, -1.6930, -6.2832, -0.4700, -0.1960, -0.1740, -0.2270,
                            0.2630, -0.1050, -0.1890, -0.1620, -0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270]).to(self.device)}
        
        q = (q - limits['lower']) / (limits['upper'] - limits['lower'])

        return 2*q - 1     
    
    def _object_start_pose(self, allegro_pose, table_pose_dy, table_pose_dz):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_pose.p.x

        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.13 #small change; take note

        object_start_pose.p.y = allegro_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_pose.p.z + pose_dz  # + 1000.0 #small change; take note

        return object_start_pose


    def _load_main_object_asset(self):
        """Load manipulated object and goal assets."""
        object_asset_options = gymapi.AssetOptions()
        object_assets = []
        for object_asset_file in self.asset_files_dict:
            object_asset_dir = os.path.dirname(object_asset_file['urdf'])
            object_asset_fname = os.path.basename(object_asset_file['urdf'])
            object_asset_ = self.gym.load_asset(self.sim, object_asset_dir, object_asset_fname, object_asset_options)
            object_assets.append(object_asset_)
        object_rb_count = self.gym.get_asset_rigid_body_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        object_shapes_count = self.gym.get_asset_rigid_shape_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        return object_assets, object_rb_count, object_shapes_count


    def _extra_reset_rules(self, resets):


        if self.cfg["env"]["resetOnArmCollision"]:
            resets = resets | (self.hand_forces > self.cfg["env"]["ArmTableCollisionThreshold"])

        if self.cfg["env"]["resetOnCollision"]:

            resets = resets | ( torch.max(self.contact_forces, dim=-1).values > self.cfg["env"]["ContactForceThreshold"])
        
        if self.cfg["env"]["resetOnFingerCrash"]:
            
            fingertip_table_dist = torch.min(self.finger_pos[:,:,-1] - self.table_pose.p.z, dim=-1).values
            resets = resets | (fingertip_table_dist < self.cfg["env"]["FingerClearanceThreshold"])

        return resets


    def _object_keypoint_offsets(self):
        """Regrasping task uses only a single object keypoint since we do not care about object orientation."""
        return [[0, 0, 0]]
    

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return [self.goal_object_indices[env_ids]]


    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2 (same as in allegro_hand.py)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        print("GROUND ADDED TO SIM")


    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")

        object_asset_root = asset_root

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.hand_arm_asset_file} from {asset_root}")
        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)
        print(f"Loaded asset {allegro_kuka_asset}")


        self.hand_joint_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.hand_joints
        ]

        self.cabinet_handle_mesh = trimesh.load(os.path.join(self.asset_root, "sektion_cabinet_model/meshes/drawer_handle.obj"))
        self.cabinet_handle_pc = np.array(trimesh.sample.sample_surface(self.cabinet_handle_mesh, self.point_cloud_sampled_dim, seed=0)[0]) #no notion of scale


        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert (
            self.num_hand_arm_dofs == num_hand_arm_dofs
        ), f"Number of DOFs in asset {allegro_kuka_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes

        allegro_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(allegro_kuka_asset, i) for i in range(self.num_hand_arm_bodies)
        ]
        self.joint_names = self.gym.get_asset_joint_names(allegro_kuka_asset)
        self.link_names = allegro_rigid_body_names
        # get link handles
        self.link_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in allegro_rigid_body_names
        ]

        print(f"Allegro num rigid bodies: {self.num_hand_arm_bodies}")
        print(f"Allegro rigid bodies: {allegro_rigid_body_names}")

        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.allegro_sensors = []
        allegro_sensor_pose = gymapi.Transform()

        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(allegro_hand_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(allegro_hand_dof_props["upper"][i])

        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)

        allegro_pose = gymapi.Transform()
        allegro_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.45, 0)

        allegro_pose.r = gymapi.Quat().from_euler_zyx(0, 0.0, -1.5708)

        #What is this used for??
        additional_rb, additional_shapes = self._load_additional_assets(object_asset_root, allegro_pose)
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes


        self.allegro_hands = []
        self.envs = []

        self.allegro_hand_indices = []
        self.object_handles = []
        self.rgb_tensors = []
        self.depth_tensors = []
        self.seg_tensors = []
        self.cams = []
        self.object_names = []
        self.cabinet_object_indices = []
        self.cabinet_handle_point_clouds = []
        self.allegro_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]
        self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "link7")

        #check where this is used. Probably for putting penalty on table collision. 
        self.hand_handles = [self.gym.find_asset_rigid_body_index(allegro_kuka_asset, f"link{i}") for i in range(2,7)]
        self.allegro_handles = [self.gym.find_asset_rigid_body_index(allegro_kuka_asset, f"link_{i}.0") for i in range(15)]

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies*100, max_agg_shapes*100, True)

            allegro_actor = self.gym.create_actor(env_ptr, allegro_kuka_asset, allegro_pose, "allegro", i, -1, 0)

            populate_dof_properties(allegro_hand_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_hand_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.allegro_hand_indices.append(allegro_hand_idx)


            cabinet_handle = self.gym.create_actor(
            env_ptr, self.cabinet_asset, self.cabinet_pose, "cabinet_object", i, -1, 1)
            self.cabinet_object_idx = self.gym.get_actor_index(env_ptr, cabinet_handle, gymapi.DOMAIN_SIM)
            self.cabinet_object_indices.append(self.cabinet_object_idx)
            self.cabinet_handle_point_clouds.append(self.cabinet_handle_pc)
            
            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_actor)

            self.gym.end_aggregate(env_ptr)

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.link_handles = to_torch(self.link_handles, dtype=torch.long, device=self.device)
        self.hand_joint_handles = to_torch(self.hand_joint_handles, dtype=torch.long, device=self.device)
        self.hand_handles = to_torch(self.hand_handles, dtype=torch.long, device=self.device)
        # self.hand_joint_point_clouds = to_torch(hand_joint_point_clouds, dtype=torch.float, device=self.device)



        self.allegro_hand_indices = to_torch(self.allegro_hand_indices, dtype=torch.long, device=self.device)
        self.cabinet_object_indices = to_torch(self.cabinet_object_indices, dtype=torch.long, device=self.device)
        self.cabinet_handle_point_clouds = to_torch(self.cabinet_handle_point_clouds, dtype=torch.float, device=self.device)

        self.rigid_body_palm_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link7")
        self.rigid_body_drawer_top_ind = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_handle, "drawer_top")

        self.rigid_body_index_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link_3.0")
        self.rigid_body_middle_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link_7.0")
        self.rigid_body_ring_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link_11.0")
        self.rigid_body_thumb_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link_15.0")

        self.env_kuka_ind = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_ENV)
        self.env_cabinet_ind = self.gym.get_actor_index(env_ptr, cabinet_handle, gymapi.DOMAIN_ENV)
        

        allegro_rigid_body_names = self.gym.get_actor_rigid_body_names( env_ptr, allegro_actor)
        allegro_arm_body_names = [f"link{i}" for i in range(1, 8)]

        self.rigid_body_arm_inds = torch.zeros(len(allegro_arm_body_names), dtype=torch.long, device=self.device)
        for i, n in enumerate(allegro_arm_body_names):
            self.rigid_body_arm_inds[i] = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, n)

        self.init_grasp_pose()

    def init_grasp_pose(self):
        self.local_finger_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.local_finger_grasp_pos[:, 0] = 0.045
        self.local_finger_grasp_pos[:, 1] = 0.01
        self.local_finger_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.local_finger_grasp_rot[:, 3] = 1.0

        self.index_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.index_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.index_grasp_rot[..., 3] = 1.0

        self.middle_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.middle_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.middle_grasp_rot[..., 3] = 1.0

        self.ring_grasp_pos = torch.zeros_like(self.local_finger_grasp_pos)
        self.ring_grasp_rot = torch.zeros_like(self.local_finger_grasp_rot)
        self.ring_grasp_rot[..., 3] = 1.0

        self.local_thumb_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.local_thumb_grasp_pos[:, 0] = 0.06
        self.local_thumb_grasp_pos[:, 1] = 0.01
        self.local_thumb_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.local_thumb_grasp_rot[:, 3] = 1.0

        self.thumb_grasp_pos = torch.zeros_like(self.local_thumb_grasp_pos)
        self.thumb_grasp_rot = torch.zeros_like(self.local_thumb_grasp_rot)
        self.thumb_grasp_rot[..., 3] = 1.0

        self.drawer_local_grasp_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.drawer_local_grasp_pos[:, 0] = 0.305
        self.drawer_local_grasp_pos[:, 2] = 0.01
        self.drawer_local_grasp_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.drawer_local_grasp_rot[..., 3] = 1.0

        self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        self.drawer_grasp_rot[..., 3] = 1.0


    def _action_penalties(self) -> Tuple[Tensor, Tensor]:
        kuka_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 0:7]), dim=-1) * self.kuka_actions_penalty_scale
        )
        allegro_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 7 : self.num_hand_arm_dofs]), dim=-1)
            * self.allegro_actions_penalty_scale
        )

        return -1 * kuka_actions_penalty, -1 * allegro_actions_penalty


    def _compute_resets(self, is_success):
        
        if self.max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            # self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= self.max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(resets), resets)
        resets = self._extra_reset_rules(resets)
        return resets


    def compute_kuka_reward(
        self,
        reset_buf: Tensor, progress_buf: Tensor, successes: Tensor, actions: Tensor,
        index_grasp_pos: Tensor, middle_grasp_pos: Tensor, ring_grasp_pos: Tensor, thumb_grasp_pos: Tensor,
        drawer_grasp_pos: Tensor, to_goal: Tensor,
        finger_dist_reward_scale: float, thumb_dist_reward_scale: float, around_handle_reward_scale: float,
        open_bonus_reward_scale: float, goal_dist_reward_scale: float, open_pose_reward_scale: float,
        goal_bonus_reward_scale: float, action_penalty_scale: float,
        contact_forces: Tensor, arm_inds: Tensor, max_episode_length: int
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Index to handle distance
        ih_d = torch.norm(drawer_grasp_pos - index_grasp_pos, p=2, dim=-1)
        ih_d = torch.clamp(ih_d, min=0.008)
        ih_dist_reward = 1.0 / (0.04 + ih_d)

        # Middle to object distance
        mh_d = torch.norm(drawer_grasp_pos - middle_grasp_pos, p=2, dim=-1)
        mh_d = torch.clamp(mh_d, min=0.008)
        mh_dist_reward = 1.0 / (0.04 + mh_d)

        # Ring to object distance
        rh_d = torch.norm(drawer_grasp_pos - ring_grasp_pos, p=2, dim=-1)
        rh_d = torch.clamp(rh_d, min=0.008)
        rh_dist_reward = 1.0 / (0.04 + rh_d)

        # Thumb to object distance
        th_d = torch.norm(drawer_grasp_pos - thumb_grasp_pos, p=2, dim=-1)
        th_d = torch.clamp(th_d, min=0.008)
        th_dist_reward = 1.0 / (0.04 + th_d)

        # Around handle reward
        ih_z_dist = index_grasp_pos[:, 2] - drawer_grasp_pos[:, 2]
        th_z_dist = drawer_grasp_pos[:, 2] - thumb_grasp_pos[:, 2]
        around_handle_reward = torch.zeros_like(ih_dist_reward)
        around_handle_reward = torch.where(ih_z_dist * th_z_dist > 0, around_handle_reward + 0.5, around_handle_reward)

        # Regularization on the actions
        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Drawer is opened
        drawer_opened = (0.4 - to_goal) > 0.01

        # Drawer open bonus
        open_bonus_reward = torch.zeros_like(ih_dist_reward)
        open_bonus_reward = torch.where(drawer_opened, open_bonus_reward + 0.5, open_bonus_reward)

        # Drawer open reward
        goal_reward = 0.4 - to_goal

        # Fingers leave handle while drawer is opened
        fingers_leave_handle = torch.zeros_like(to_goal)
        fingers_leave_handle = torch.where(ih_d >= 0.04, torch.ones_like(fingers_leave_handle), fingers_leave_handle)
        fingers_leave_handle = torch.where(th_d >= 0.04, torch.ones_like(fingers_leave_handle), fingers_leave_handle)

        # Correct open pose reward
        open_pose_reward = around_handle_reward * (1.0 - fingers_leave_handle) * goal_reward * 2.0

        # Bonus if drawer is fully opened
        goal_bonus_reward = torch.zeros_like(to_goal)
        goal_bonus_reward = torch.where(to_goal <= 0.1, goal_bonus_reward + 0.5, goal_bonus_reward)

        self.rewards_episode["raw_ih_dist_reward"] += ih_dist_reward
        self.rewards_episode["raw_mh_dist_reward"] += mh_dist_reward
        self.rewards_episode["raw_rh_dist_reward"] += rh_dist_reward
        self.rewards_episode["raw_th_dist_reward"] += th_dist_reward
        self.rewards_episode["raw_around_handle_reward"] += around_handle_reward
        self.rewards_episode["raw_open_bonus_reward"] += open_bonus_reward
        self.rewards_episode["raw_goal_reward"] += goal_reward
        self.rewards_episode["raw_open_pose_reward"] += open_pose_reward
        self.rewards_episode["raw_goal_bonus_reward"] += goal_bonus_reward
        self.rewards_episode["raw_action_penalty"] += -1*action_penalty

        # Total reward
        rewards = finger_dist_reward_scale * ih_dist_reward \
            + finger_dist_reward_scale * mh_dist_reward \
            + finger_dist_reward_scale * rh_dist_reward \
            + thumb_dist_reward_scale * th_dist_reward \
            + around_handle_reward_scale * around_handle_reward \
            + open_bonus_reward_scale * open_bonus_reward \
            + goal_dist_reward_scale * goal_reward \
            + open_pose_reward_scale * open_pose_reward \
            + goal_bonus_reward_scale * goal_bonus_reward \
            - action_penalty_scale * action_penalty
        
        # Goal reached and hand cannot leave handle
        s = torch.where(successes < 10.0, torch.zeros_like(successes), successes)
        successes = torch.where(goal_reward >= 0.4 * 0.8, torch.ones_like(successes) + successes, s)

        # Arm collision
        arm_collision = torch.any(torch.norm(contact_forces[:, arm_inds, :], dim=2) > 1.0, dim=1)
        reset_buf = torch.where(arm_collision, torch.ones_like(reset_buf), reset_buf)

        # Max episode length exceeded
        reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

        binary_s = torch.where(successes >= 10, torch.ones_like(successes), torch.zeros_like(successes))
        successes = torch.where(reset_buf > 0, binary_s, successes)

        
        self.rewards_episode["ih_dist_reward"] += ih_dist_reward*finger_dist_reward_scale
        self.rewards_episode["mh_dist_reward"] += mh_dist_reward*finger_dist_reward_scale
        self.rewards_episode["rh_dist_reward"] += rh_dist_reward*finger_dist_reward_scale
        self.rewards_episode["th_dist_reward"] += th_dist_reward*thumb_dist_reward_scale
        self.rewards_episode["around_handle_reward"] += around_handle_reward*around_handle_reward_scale
        self.rewards_episode["open_bonus_reward"] += open_bonus_reward*open_bonus_reward_scale
        self.rewards_episode["goal_reward"] += goal_reward*goal_dist_reward_scale
        self.rewards_episode["open_pose_reward"] += open_pose_reward*open_pose_reward_scale
        self.rewards_episode["goal_bonus_reward"] += goal_bonus_reward*goal_bonus_reward_scale
        self.rewards_episode["action_penalty"] += -1*action_penalty*action_penalty_scale

        self.extras["rewards_episode"] = self.rewards_episode

        return rewards, reset_buf, successes


    def _eval_stats(self, is_success: Tensor) -> None:
        if self.eval_stats:
            frame: int = self.frame_since_restart
            n_frames = torch.empty_like(self.last_success_step).fill_(frame)
            self.success_time = torch.where(is_success, n_frames - self.last_success_step, self.success_time)
            self.last_success_step = torch.where(is_success, n_frames, self.last_success_step)
            mask_ = self.success_time > 0
            if any(mask_):
                avg_time_mean = ((self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)).item()
            else:
                avg_time_mean = math.nan

            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            self.total_num_resets += self.reset_buf

            reset_ids = self.reset_buf.nonzero().squeeze()
            last_successes = self.successes[reset_ids].long()
            self.successes_count[last_successes] += 1

            if frame % 100 == 0:
                # The direct average shows the overall result more quickly, but slightly undershoots long term
                # policy performance.
                print(f"Max num successes: {self.successes.max().item()}")
                print(f"Average consecutive successes: {self.prev_episode_successes.mean().item():.2f}")
                print(f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}")
                print(f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}")
                print(f"Last ep successes: {self.prev_episode_successes.mean().item():.2f}")
                # print(f"Last ep true objective: {self.prev_episode_true_objective.mean().item():.2f}")

                self.eval_summaries.add_scalar("last_ep_successes", self.prev_episode_successes.mean().item(), frame)
                # self.eval_summaries.add_scalar(
                #     "last_ep_true_objective", self.prev_episode_true_objective.mean().item(), frame
                # )
                self.eval_summaries.add_scalar(
                    "reset_stats/reset_percentage", (self.total_num_resets > 0).sum() / self.num_envs, frame
                )
                self.eval_summaries.add_scalar("reset_stats/min_num_resets", self.total_num_resets.min().item(), frame)

                self.eval_summaries.add_scalar("policy_speed/avg_success_time_frames", avg_time_mean, frame)
                frame_time = self.control_freq_inv * self.dt
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_time_seconds", avg_time_mean * frame_time, frame
                )
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_per_minute", 60.0 / (avg_time_mean * frame_time), frame
                )
                print(f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}")

                # create a matplotlib bar chart of the self.successes_count
                import matplotlib.pyplot as plt

                plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
                plt.title("Successes histogram")
                plt.xlabel("Successes")
                plt.ylabel("Frequency")
                plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
                plt.clf()

    def compute_observations(self) -> Tuple[Tensor, int]:

        #refreshing is important to get correct values
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.index_grasp_rot[:], self.index_grasp_pos[:] = tf_combine(
            self.index_rot, self.index_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.middle_grasp_rot[:], self.middle_grasp_pos[:] = tf_combine(
            self.middle_rot, self.middle_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.ring_grasp_rot[:], self.ring_grasp_pos[:] = tf_combine(
            self.ring_rot, self.ring_pos,
            self.local_finger_grasp_rot, self.local_finger_grasp_pos
        )
        self.thumb_grasp_rot[:], self.thumb_grasp_pos[:] = tf_combine(
            self.thumb_rot, self.thumb_pos,
            self.local_thumb_grasp_rot, self.local_thumb_grasp_pos
        )

        self.drawer_grasp_rot[:], self.drawer_grasp_pos[:] = tf_combine(
            self.drawer_handle_rot, self.drawer_handle_pos,
            self.drawer_local_grasp_rot, self.drawer_local_grasp_pos
        )

        self.index_to_handle[:] = self.drawer_grasp_pos - self.index_grasp_pos
        self.middle_to_handle[:] = self.drawer_grasp_pos - self.middle_grasp_pos
        self.ring_to_handle[:] = self.drawer_grasp_pos - self.ring_grasp_pos
        self.thumb_to_handle[:] = self.drawer_grasp_pos - self.thumb_grasp_pos

        self.point_cloud_buf = quat_apply(
                self.drawer_handle_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.cabinet_handle_point_clouds
            ) + self.drawer_handle_pos[:, None]

        self.to_goal[:] = 0.4 - self.cabinet_dof_pos[:, 3].unsqueeze(1) # 0.4 is the maximum distance

        self.kuka_dof_pos_scaled[:] = \
            (2.0 * (self.arm_hand_dof_pos - self.arm_hand_dof_lower_limits) /
                (self.arm_hand_dof_upper_limits - self.arm_hand_dof_lower_limits) - 1.0)
        self.kuka_dof_vel_scaled[:] = self.kuka_dof_vel * self.hand_dof_speed_scale #check this

        return 


    def capture_depth(self):
        
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        # print(self.depth_tensors[0].device)
        # rgb = torch.stack(self.rgb_tensors).to(self.device)
        depth = torch.stack(self.depth_tensors).to(self.device)
        seg = torch.stack(self.seg_tensors).to(self.device)
        self.gym.end_access_image_tensors(self.sim)
        # process raw data
        # 1. filter out nan/inf data of depth
        depth = -1*depth

        depth[depth > 1] = 1
        depth[depth < 0.05] = 0.05

        depth = depth.unsqueeze(1)
        seg = seg.unsqueeze(1)

        #boolean tensor of size of seg with 10% of True values
        mask = torch.rand((5,5)).to(self.device) < 0.05
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(depth.shape[-2], depth.shape[-1]), mode='nearest')
        mask = mask.int().bool()
        depth[mask*(seg==2)] = 1.0

        self.blur = T.GaussianBlur(kernel_size=59, sigma=2.0)
        depth = self.blur(depth)*(seg == 2) + depth*(seg != 2)

        # # small_tensor = F.interpolate(depth, scale_factor=0.5, mode='nearest')
        # # # Upscale back to the original size
        # # blurred_depth = F.interpolate(small_tensor, size=(depth.shape[-2], depth.shape[-1]), mode='nearest')

        # depth = blurred_depth
        #depth = 
        z  = batch_rotate_images_torch(depth, self.cam_rnd_R)
        depth = z.unsqueeze(1) 

        return depth


    def compute_task_buffers(self):

        if self.enable_camera_sensors:
            
            if self.enable_depth_camera:
                #not implemented in this branch
                depth = self.capture_depth()

                if self.rgbd_camera_height != depth.shape[-1]:
                    depth = depth_img_resize(depth, size=self.rgbd_camera_height)
                    
                at_reset_env_ids = torch.where(self.reset_buf == 1)[0]
                
                d = depth.unsqueeze(1).repeat(1, self.depth_buf.shape[1], 1, 1, 1).clone() 
                self.depth_buf[at_reset_env_ids] = d[at_reset_env_ids]

                not_at_reset_env_ids = torch.where(self.reset_buf == 0)[0]
                self.depth_buf[not_at_reset_env_ids, :] = torch.cat([
                    self.depth_buf[not_at_reset_env_ids, 1:],
                    depth[not_at_reset_env_ids].unsqueeze(1),
                ], dim=1)
        

            
        if self.enable_proprio_history:    
            
            at_reset_env_ids = torch.where(self.reset_buf == 1)[0]
            # temp = self.arm_hand_dof_pos # self.obs_buf #
            if self.use_obs_as_prop:
                temp = self.obs_buf #self.arm_hand_dof_pos
            else:
                temp = self.arm_hand_dof_pos 
                #temp = self.scale_q(self.arm_hand_dof_pos)

            # Get proprio for the reset envs
            proprio_history = temp[at_reset_env_ids]
            # Repeat the history to fill the buffer
            proprio_history = proprio_history.unsqueeze(1).repeat(1,self.proprio_buf.shape[1],1).clone()
            # Update the buffer
            self.proprio_buf[at_reset_env_ids] = proprio_history

            not_at_reset_env_ids = torch.where(self.reset_buf == 0)[0]
            # Shift the buffer and add the new observation
            self.proprio_buf[not_at_reset_env_ids, :] = torch.cat([
                self.proprio_buf[not_at_reset_env_ids, 1:],
                temp[not_at_reset_env_ids].unsqueeze(1),
            ], dim=1)

        
        if self.enable_point_cloud:
            at_reset_env_ids = torch.where(self.reset_buf == 1)[0]
            point_cloud = self.point_cloud_buf[at_reset_env_ids]
            point_cloud = point_cloud.unsqueeze(1).repeat(1,self.pc_buf.shape[1],1,1).clone()
            self.pc_buf[at_reset_env_ids] = point_cloud

            not_at_reset_env_ids = torch.where(self.reset_buf == 0)[0]
            self.pc_buf[not_at_reset_env_ids, :] = torch.cat([
                self.pc_buf[not_at_reset_env_ids, 1:],
                self.point_cloud_buf[not_at_reset_env_ids].unsqueeze(1),
            ], dim=1)
        

        if self.enable_action_history:

            at_reset_env_ids = torch.where(self.reset_buf == 1)[0]
            # TODO: actions should be unscaled
            action_history = self.actions[at_reset_env_ids]
            action_history = action_history.unsqueeze(1).repeat(1,self.action_buf.shape[1],1).clone()
            self.action_buf[at_reset_env_ids] = action_history

            not_at_reset_env_ids = torch.where(self.reset_buf == 0)[0]
            self.action_buf[not_at_reset_env_ids, :] = torch.cat([
                self.action_buf[not_at_reset_env_ids, 1:],
                self.actions[not_at_reset_env_ids].unsqueeze(1),
            ], dim=1)

        if self.enable_attn_mask:
            
            at_reset_env_ids = torch.where(self.reset_buf == 1)[0]
            # Attn mask size: (num_envs, stage2_hist_len)
            self.attn_mask[at_reset_env_ids] = torch.zeros_like(self.attn_mask[at_reset_env_ids])
            # TODO: Change here for time shifted attention mask
            self.attn_mask[at_reset_env_ids,-1] = 1 

            not_at_reset_env_ids = torch.where(self.reset_buf == 0)[0]
            self.attn_mask[not_at_reset_env_ids] = torch.cat([
                self.attn_mask[not_at_reset_env_ids, 1:],
                torch.ones_like(self.attn_mask[not_at_reset_env_ids,-1]).unsqueeze(1),
            ], dim=1)

            self.timesteps = torch.cumsum(self.attn_mask, dim=1) - 1 
            self.timesteps[self.timesteps < 0] = self.stage2_hist_len #pad token
        

    def compute_full_state(self):
        self.obs_buf[:] = torch.cat([
            self.kuka_dof_pos_scaled, self.kuka_dof_vel_scaled,
            self.palm_pos,
            self.index_grasp_pos, self.middle_grasp_pos, self.ring_grasp_pos, self.thumb_grasp_pos,
            self.drawer_grasp_pos,
            self.index_to_handle, self.middle_to_handle, self.ring_to_handle, self.thumb_to_handle,
            self.to_goal
        ], dim=-1)


    def clamp_obs(self, obs_buf: Tensor) -> None:
        if self.clamp_abs_observations > 0:
            obs_buf.clamp_(-self.clamp_abs_observations, self.clamp_abs_observations)

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot

    def reset_target_pose(self, env_ids: Tensor) -> None:
        self._reset_target(env_ids)

        self.reset_goal_buf[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.closest_keypoint_max_dist[env_ids] = -1

    def reset_object_pose(self, env_ids):

        # since we reset the object, we also should update distances between fingers and the object
        self.closest_fingertip_dist[env_ids] = -1
        self.closest_joint_tgt_rel_pose_dist[env_ids] = -1 
        self.closest_fingertip_shape_dist[env_ids] = -1
        self.closest_palm_dist[env_ids] = -1
        self.furthest_hand_dist[env_ids] = -1

    def deferred_set_actor_root_state_tensor_indexed(self, obj_indices: List[Tensor]) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            # nothing to set
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def reset_idx(self, env_ids: Tensor) -> None:

        self.reset_target_pose(env_ids)


        # reset allegro hand
        delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
        delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos

        rand_dof_floats = torch_rand_float(0.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)

        rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

        noise_coeff = torch.zeros_like(self.hand_arm_default_dof_pos, device=self.device)

        noise_coeff[0:7] = self.reset_dof_pos_noise_arm
        noise_coeff[7 : self.num_hand_arm_dofs] = self.reset_dof_pos_noise_fingers

        allegro_pos = self.hand_arm_default_dof_pos + noise_coeff * rand_delta

        self.arm_hand_dof_pos[env_ids, :] = allegro_pos

        rand_vel_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)
        self.arm_hand_dof_vel[env_ids, :] =self.reset_dof_vel_noise * rand_vel_floats

        self.prev_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos
        self.cur_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos

        self.prev_targets[env_ids, self.num_hand_arm_dofs:] = 0.0
        self.cur_targets[env_ids, self.num_hand_arm_dofs:] = 0.0


        success_state = self.gym.set_dof_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_state)
        )

        success_target = self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets)
        )

        assert success_state and success_target

        self.progress_buf[env_ids] = 0
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        # self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        # self.true_objective[env_ids] = 0
        self.lifted_object[env_ids] = False

        # -1 here indicates that the value is not initialized
        self.closest_keypoint_max_dist[env_ids] = -1
        self.closest_fingertip_dist[env_ids] = -1
        self.furthest_hand_dist[env_ids] = -1
        self.closest_palm_dist[env_ids] = -1
        self.closest_fingertip_shape_dist[env_ids] = -1
        self.closest_joint_tgt_rel_pose_dist[env_ids] = -1

        self.near_goal_steps[env_ids] = 0

        for key in self.rewards_episode.keys():
            self.rewards_episode[key][env_ids] = 0
        
        self.extras["successes"] = self.prev_episode_successes.mean()
        self.extras["scalars"] = dict()

    def pre_physics_step(self, actions):

        self.reset_buf[:] = 0

        self.actions = actions.clone().to(self.device)

        if self.privileged_actions:
            torque_actions = actions[:, :3]
            actions = actions[:, 3:]

        if self.use_relative_control:
            raise NotImplementedError("Use relative control False for now")
        else:
            if self.old_action_space:
                # target position control for the hand DOFs
                self.cur_targets[:, 7 : self.num_hand_arm_dofs] = scale(
                    actions[:, 7 : self.num_hand_arm_dofs],
                    self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                    self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
                )
                self.cur_targets[:, 7 : self.num_hand_arm_dofs] = (
                    self.act_moving_average * self.cur_targets[:, 7 : self.num_hand_arm_dofs]
                    + (1.0 - self.act_moving_average) * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
                )
                self.cur_targets[:, 7 : self.num_hand_arm_dofs] = tensor_clamp(
                    self.cur_targets[:, 7 : self.num_hand_arm_dofs],
                    self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                    self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
                )

                targets = self.prev_targets[:, :7] + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
                self.cur_targets[:, :7] = tensor_clamp(
                    targets, self.arm_hand_dof_lower_limits[:7], self.arm_hand_dof_upper_limits[:7]
                ) 

                self.prev_targets[:, :] = self.cur_targets[:, :]

            else:
                # targets = self.prev_targets + self.hand_dof_speed_scale * self.dt * self.actions
                #unscaled for now, may change later
                self.cur_targets[:,:self.num_hand_arm_dofs] = scale(
                    self.actions,
                    self.arm_hand_dof_lower_limits,
                    self.arm_hand_dof_upper_limits,
                )

                # Limits the first 7 joints to a maximum of 1/120 rad change with respect to the previous targets
                if self.limit_arm_delta_target:
                    self.cur_targets[:, :7] = tensor_clamp(self.cur_targets[:, :7],
                                                           self.prev_targets[:, :7] - self.hand_dof_speed_scale*self.dt,
                                                           self.prev_targets[:, :7] + self.hand_dof_speed_scale*self.dt)

                self.cur_targets[:,:self.num_hand_arm_dofs] = tensor_clamp(
                    self.cur_targets[:,:self.num_hand_arm_dofs], self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
                )

            
        self.prev_targets[:, :] = self.cur_targets[:, :] 

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # apply torques
        if self.privileged_actions:
            torque_actions = torque_actions.unsqueeze(1)
            torque_amount = self.privileged_actions_torque
            torque_actions *= torque_amount
            self.action_torques[:, self.object_rb_handles, :] = torque_actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim, None, gymtorch.unwrap_tensor(self.action_torques), gymapi.ENV_SPACE
            )

    def post_physics_step(self):


        self.frame_since_restart += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()

        self.rew_buf[:], self.reset_buf[:], self.successes[:] = self.compute_kuka_reward(
            self.reset_buf, self.progress_buf, self.successes, self.actions,
            self.index_grasp_pos, self.middle_grasp_pos, self.ring_grasp_pos, self.thumb_grasp_pos,
            self.drawer_grasp_pos, self.to_goal.squeeze(1),
            self.finger_dist_reward_scale, self.thumb_dist_reward_scale, self.around_handle_reward_scale,
            self.open_bonus_reward_scale, self.goal_dist_reward_scale, self.open_pose_reward_scale,
            self.goal_bonus_reward_scale, self.action_penalty_scale,
            self.contact_forces, self.rigid_body_arm_inds, self.max_episode_length
        )

        #reset should go here along with a refresh 

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.set_actor_root_state_tensor_indexed()

        if len(reset_env_ids) > 0:
            self.compute_observations()

        self.compute_full_state()


        self.clamp_obs(self.obs_buf)

        self.compute_task_buffers()

        self._eval_stats(self.successes)


        if self.viewer and self.debug_viz:
            # draw axes on target object

            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            axes_geom = gymutil.AxesGeometry(0.1)

            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.003, 8, 8, sphere_pose, color=(1, 1, 0))
            sphere_geom_white = gymutil.WireframeSphereGeometry(0.03, 8, 8, sphere_pose, color=(1, 1, 1))
            sphere_geom_red = gymutil.WireframeSphereGeometry(0.03, 8, 8, sphere_pose, color=(1, 0, 0))
            sphere_geom_purple = gymutil.WireframeSphereGeometry(0.03, 8, 8, sphere_pose, color=(1, 0, 1))
            sphere_geom_yellow = gymutil.WireframeSphereGeometry(0.03, 8, 8, sphere_pose, color=(1, 1, 0))

            #make dots at cam_pos and cam_target 
            # for i in range(self.num_envs):
            #     cam_pos_transform = gymapi.Transform()
            #     cam_pos_transform.p = self.cam_pos[1]
            #     gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], cam_pos_transform)
            #     cam_tgt_transform = gymapi.Transform()
            #     cam_tgt_transform.p = self.cam_target[1]
            #     gymutil.draw_lines(sphere_geom_purple, self.gym, self.viewer, self.envs[i], cam_tgt_transform)

            # palm_center_pos_cpu = self.palm_center_pos.cpu().numpy()
            # palm_rot_cpu = self._palm_rot.cpu().numpy()
            # for i in range(self.num_envs):
            #     palm_center_transform = gymapi.Transform()
            #     palm_center_transform.p = gymapi.Vec3(*palm_center_pos_cpu[i])
            #     palm_center_transform.r = gymapi.Quat(*palm_rot_cpu[i])
            #     gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], palm_center_transform)

            goal_pos = self.thumb_pos.cpu().numpy()
            for i in range(self.num_envs):
                goal_transform = gymapi.Transform()
                goal_transform.p = gymapi.Vec3(*goal_pos[i])
                gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], goal_transform)

            goal_pos = self.index_pos.cpu().numpy()
            for i in range(self.num_envs):
                goal_transform = gymapi.Transform()
                goal_transform.p = gymapi.Vec3(*goal_pos[i])
                gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], goal_transform)

            goal_pos = self.middle_pos.cpu().numpy()
            for i in range(self.num_envs):
                goal_transform = gymapi.Transform()
                goal_transform.p = gymapi.Vec3(*goal_pos[i])
                gymutil.draw_lines(sphere_geom_purple, self.gym, self.viewer, self.envs[i], goal_transform)

            goal_pos = self.ring_pos.cpu().numpy()
            for i in range(self.num_envs):
                goal_transform = gymapi.Transform()
                goal_transform.p = gymapi.Vec3(*goal_pos[i])
                gymutil.draw_lines(sphere_geom_yellow, self.gym, self.viewer, self.envs[i], goal_transform)


            # limit_z = self.table_pose.p.z
            # for i in range(self.num_envs):
            #     goal_transform = gymapi.Transform()
            #     goal_transform.p = gymapi.Vec3(0,0,limit_z)
            #     gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], goal_transform)


            # for i in range(self.num_envs):
            #     origin_transform = gymapi.Transform()
            #     origin_transform.p = gymapi.Vec3(0, 0, 0)
            #     gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], origin_transform)

            # cam_pos = np.array(self.cam_loc_p)
            # for i in range(self.num_envs):
            #     goal_transform = gymapi.Transform()
            #     goal_transform.p = gymapi.Vec3(*cam_pos)
            #     gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], goal_transform)
            # for j in range(self.num_allegro_fingertips):
            #     fingertip_pos_cpu = self.fingertip_pos[:, j].cpu().numpy()
            #     fingertip_rot_cpu = self.fingertip_rot[:, j].cpu().numpy()

            #     for i in range(self.num_envs):
            #         fingertip_transform = gymapi.Transform()
            #         fingertip_transform.p = gymapi.Vec3(*fingertip_pos_cpu[i])
            #         # fingertip_transform.r = gymapi.Quat(*fingertip_rot_cpu[i])

            #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], fingertip_transform)

            # for j in range(self.num_keypoints):
            #     keypoint_pos_cpu = self.obj_keypoint_pos[:, j].cpu().numpy()
            #     goal_keypoint_pos_cpu = self.goal_keypoint_pos[:, j].cpu().numpy()

            #     for i in range(self.num_envs):
            #         keypoint_transform = gymapi.Transform()
            #         keypoint_transform.p = gymapi.Vec3(*keypoint_pos_cpu[i])
            #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], keypoint_transform)

            #         goal_keypoint_transform = gymapi.Transform()
            #         goal_keypoint_transform.p = gymapi.Vec3(*goal_keypoint_pos_cpu[i])
            #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], goal_keypoint_transform)
            # 
            # # #draw point cloud

            # if self.traj_pc is not None:
            for j in range(self.point_cloud_sampled_dim):
                # point_cloud_pos_cpu = self.point_cloud_buf[:,j].cpu().numpy()
                point_cloud_pos_cpu = self.point_cloud_buf[:,j].cpu().numpy()
                for i in range(self.num_envs):
                    point_cloud_transform = gymapi.Transform()
                    point_cloud_transform.p = gymapi.Vec3(*point_cloud_pos_cpu[i])
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], point_cloud_transform)
        

            # # draw hand joint point cloud
            # for k in range(len(self.hand_joint_handles)):
            #     for j in range(self.point_cloud_sampled_dim):
            #         hand_joint_point_cloud_pos_cpu = self.hand_joint_point_cloud_buf[:,k,j].cpu().numpy()
            #         for i in range(self.num_envs):
            #             point_cloud_transform = gymapi.Transform()
            #             point_cloud_transform.p = gymapi.Vec3(*hand_joint_point_cloud_pos_cpu[i])
            #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], point_cloud_transform)
            

            # for i in range(self.num_envs):
            #     for j in range(4):
            #         hnd_pos_transform = gymapi.Transform()
            #         hnd_pos_transform.p = gymapi.Vec3(*self.hnd_tgt[i][j])
            #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], hnd_pos_transform)

            # for i in range(self.num_envs):
            #     for j in range(4):
            #         hnd_pos_transform = gymapi.Transform()
            #         hnd_pos_transform.p = gymapi.Vec3(*self.fingertip_pos[i][j])
            #         gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], hnd_pos_transform)

            # object_pos_cpu = self.object_pos.cpu().numpy()

            #for i in range(self.num_envs):
            #    
            #    object_pos_transform = gymapi.Transform()
            #    object_pos_transform.p = gymapi.Vec3(*object_pos_cpu[i])
            #    gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], object_pos_transform)
            



    def _reset_target(self, env_ids: Tensor) -> None:

        # self.reset_object_pose(env_ids)

        # whether we place the bucket to the left or to the right of the table
        # left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
        # x_pos = torch.where(
        #     left_right_random > 0, 0.5 * torch.ones_like(left_right_random), -0.5 * torch.ones_like(left_right_random)
        # )
        # x_pos += torch.sign(left_right_random) * torch_rand_float(0, 0.4, (len(env_ids), 1), device=self.device)
        ####need to fix the goal position since our formulation does not take input a goal position######
        x_pos = torch_rand_float(self.cabinet_pose.p.x, self.cabinet_pose.p.x, (len(env_ids), 1), device=self.device)
        y_pos = torch_rand_float(self.cabinet_pose.p.y, self.cabinet_pose.p.y, (len(env_ids), 1), device=self.device)
        z_pos = torch_rand_float(self.cabinet_pose.p.z, self.cabinet_pose.p.z, (len(env_ids), 1), device=self.device)
        self.root_state_tensor[self.cabinet_object_indices[env_ids], 0:1] = x_pos
        self.root_state_tensor[self.cabinet_object_indices[env_ids], 1:2] = y_pos
        self.root_state_tensor[self.cabinet_object_indices[env_ids], 2:3] = z_pos

        self.cabinet_dof_state[env_ids, :] = 0.0 # reset cabinet dof state
        # we also reset the object to its initial position
        self.reset_object_pose(env_ids)

        # since we put the object back on the table, also reset the lifting reward
        self.lifted_object[env_ids] = False

        object_indices_to_reset = [self.cabinet_object_indices[env_ids],]
        self.deferred_set_actor_root_state_tensor_indexed(object_indices_to_reset)



    def _after_envs_created(self):
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
    

    def _load_additional_assets(self, object_asset_root, arm_pose):
        # goal_asset_options = gymapi.AssetOptions()
        # goal_asset_options.disable_gravity = True
        # self.goal_asset = self.gym.load_asset(
        #     self.sim, object_asset_root, self.ball_asset_file, goal_asset_options
        # )
        # goal_rb_count = self.gym.get_asset_rigid_body_count(self.goal_asset)
        # goal_shapes_count = self.gym.get_asset_rigid_shape_count(self.goal_asset)
        # return goal_rb_count, goal_shapes_count
        """
        returns: tuple (num_rigid_bodies, num_shapes)
        """
                # Load cabinet asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_options.use_mesh_materials = True
        self.cabinet_asset = self.gym.load_asset(self.sim, object_asset_root, self.cabinet_asset_file, asset_options)


        self.cabinet_pose = gymapi.Transform()
        self.cabinet_pose.p = gymapi.Vec3(0, -0.45, 0.40)
        self.cabinet_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 1.5708)
        
        cabinet_rb_count = self.gym.get_asset_rigid_body_count(self.cabinet_asset)
        cabinet_shapes_count = self.gym.get_asset_rigid_shape_count(self.cabinet_asset)
        # return goal_rb_count, goal_shapes_count
        return cabinet_rb_count, cabinet_shapes_count

    
    def start_video_recording(self):
        self.video_frames = []
        self.recording_enabled = True
    
    def stop_video_recording(self):
        self.recording_enabled = False
        return self.video_frames


    def step(self, actions: torch.Tensor, obj_pc=None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
            """Step the physics of the environment.

            Args:
                actions: actions to apply
            Returns:
                Observations, rewards, resets, info
                Observations are dict of observations (currently only one member called 'obs')
            """

            # randomize actions
            if self.dr_randomizations.get('actions', None):
                actions = self.dr_randomizations['actions']['noise_lambda'](actions)

            action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
            # apply actions
            self.pre_physics_step(action_tensor)

            # step physics and render each frame
            for i in range(self.control_freq_inv):
                if self.force_render:
                    self.render()
                self.gym.simulate(self.sim)

            # to fix!
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            # compute observatins, rewards, resets, ...cabinet_ini
            self.post_physics_step()

            #print("bucket: ", self.root_state_tensor[self.bucket_object_indices[0], :])
            
            self.control_steps += 1

            # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
            self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0) #no connection with reset?

            # randomize observations
            if self.dr_randomizations.get('observations', None):
                self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

            self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            if self.enable_depth_camera:
                self.obs_dict["depth_buf"] = self.depth_buf.to(self.rl_device)
            

            if self.enable_proprio_history:
                self.obs_dict["proprio_buf"] = self.proprio_buf.to(self.rl_device)

            if self.enable_point_cloud:
            #debug with point cloud + proprioception 15/10-1452  
                self.obs_dict["pc_buf"]= self.pc_buf.to(self.rl_device)
            
            if self.enable_action_history:
                self.obs_dict["action_buf"] = self.action_buf.to(self.rl_device)
            
            if self.cfg["env"]["input_priv"]:
                self.obs_dict["priv_buf"] = self.priv_buf.to(self.rl_device)
            
            if self.enable_attn_mask:
                self.obs_dict["attn_mask"] = self.attn_mask.to(self.rl_device)
                self.obs_dict["timesteps"] = self.timesteps.to(self.rl_device)
            
            
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

            return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


    def _extra_curriculum(self):

        #makes the reward sparser and more fine-grained over time
        self.success_tolerance, self.last_curriculum_update = tolerance_curriculum(
            self.last_curriculum_update,
            self.frame_since_restart,
            self.tolerance_curriculum_interval,
            self.prev_episode_successes,
            self.success_tolerance,
            self.initial_tolerance,
            self.target_tolerance,
            self.tolerance_curriculum_increment,
        )


    def linearize_info(self,info):
        
        z = {}
        for key in info.keys():
            if isinstance(info[key],torch.Tensor):
                z[key] = torch.mean(info[key].to(torch.float)).cpu().numpy()
            elif isinstance(info[key],float):
                z[key] = info[key]
            elif isinstance(info[key],dict):
                z.update(self.linearize_info(info[key]))
            else:
                raise NotImplementedError(f"Not implemented for type {type(info[key])}")
        
        return z 

    def _allocate_task_buffer(self):


        self.stage2_hist_len = self.cfg['env']['stage2_hist_len']
        # buffer for stage 2 vision policy distillation
        if self.enable_depth_camera and self.enable_wrist_camera:
            self.depth_buf = torch.zeros(
                (self.num_envs, self.stage2_hist_len, 2, self.rgbd_buffer_height, self.rgbd_buffer_width),
                device=self.rl_device, dtype=torch.float
            ) #1 camera for now 
            self.random_depth_noise_e = torch.zeros(
                (self.num_envs, self.rgbd_buffer_height, self.rgbd_buffer_width), device=self.device, dtype=torch.float
            )
        
        elif self.enable_depth_camera and not self.enable_wrist_camera:
            self.depth_buf = torch.zeros(
                (self.num_envs, self.stage2_hist_len, 1, self.rgbd_buffer_height, self.rgbd_buffer_width),
                device=self.rl_device, dtype=torch.float
            ) #1 camera for now 
            self.random_depth_noise_e = torch.zeros(
                (self.num_envs, self.rgbd_buffer_height, self.rgbd_buffer_width), device=self.device, dtype=torch.float
            )
        else:
            self.depth_buf = None 
        

        if self.enable_proprio_history:
            if self.use_obs_as_prop:
                sz = self.full_state_size 
            else:
                sz = 23
            self.proprio_buf = torch.zeros(
                (self.num_envs, self.stage2_hist_len, sz), device=self.rl_device, dtype=torch.float
            )
        else:
            self.proprio_buf = None
        
        if self.enable_action_history:
            self.action_buf = torch.zeros(
                (self.num_envs, self.stage2_hist_len-1, 23), device=self.rl_device, dtype=torch.float
            )
        

        if self.enable_point_cloud:
            self.pc_buf = torch.zeros((self.num_envs,self.stage2_hist_len,self.point_cloud_sampled_dim,3), device=self.rl_device, dtype=torch.float)
        else:
            self.pc_buf = None 
        
        if self.enable_attn_mask:
            self.attn_mask = torch.zeros((self.num_envs,self.stage2_hist_len),device=self.rl_device,dtype=torch.float)
        else:
            self.attn_mask = None 
        
        # if self.randomize:
        #     #for now only mass restitution and friction

        self.priv_buf = torch.zeros((self.num_envs,5),device=self.rl_device,dtype=torch.float)
    
