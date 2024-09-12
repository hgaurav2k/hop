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
class AllegroXarmThrowing(VecTask):
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

        self.simple_objects = self.cfg["env"]["doSimpleObjects"]
        self.very_simple_objects = self.cfg["env"]["doVerySimpleObjects"]
        self.dexycb_objects = self.cfg["env"]["doDexYcbObjects"]
        self.use_saved_init_pos = self.cfg["env"]["useSavedInitPose"]
        self.use_random_init_rot = self.cfg["env"]["useRandomInitRot"]
        self.use_pose_reward_unlifted = self.cfg["env"]["usePoseRewardUnlifted"]
        self.use_pose_reward_lifted = self.cfg["env"]["usePoseRewardLifted"]

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

        # self.randomize = self.cfg["task"]["randomize"] #currently set to False
        # self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomization_params = self.cfg["task"]["domain_randomization"]
        self.do_random_resets = self.cfg["task"]["do_random_resets"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.hand_joint_rew_coeff = self.cfg["env"]["handJointRewCoeff"]
        self.use_fingertip_reward = self.cfg["env"]["useFingertipReward"]
        self.use_fingertip_shape_distance_reward = self.cfg["env"]["useFingertipShapeDistReward"]
        self.use_palm_reward = self.cfg["env"]["usePalmReward"]
        self.use_hand_joint_pose_reward = self.cfg["env"]["useHandJointPoseRew"]
        self.use_lifting_reward = self.cfg["env"]["useLiftingReward"]
        self.use_keypoint_reward = self.cfg["env"]["useKeypointReward"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        # self.use_fixed_height_as_target = self.cfg["env"]["useFixedHeightAsTarget"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.allegro_actions_penalty_scale = self.cfg["env"]["allegroActionsPenaltyScale"]
        self.throw_far = self.cfg["env"]["throw_far"]
        self.bucket_in_front = self.cfg["env"]["bucket_in_front"]

        self.dof_params: DofParameters = DofParameters.from_cfg(self.cfg)

        self.initial_tolerance = self.cfg["env"]["successTolerance"]
        self.success_tolerance = self.initial_tolerance
        self.target_tolerance = self.cfg["env"]["targetSuccessTolerance"]
        self.tolerance_curriculum_increment = self.cfg["env"]["toleranceCurriculumIncrement"]
        self.tolerance_curriculum_interval = self.cfg["env"]["toleranceCurriculumInterval"]

        self.save_states = self.cfg["env"]["saveStates"]
        self.save_states_filename = self.cfg["env"]["saveStatesFile"]

        self.should_load_initial_states = self.cfg["env"]["loadInitialStates"]
        self.load_states_filename = self.cfg["env"]["loadStatesFile"]
        self.initial_root_state_tensors = self.initial_dof_state_tensors = None
        self.initial_state_idx = self.num_initial_states = 0

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"]["resetDofPosRandomIntervalFingers"]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

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

        self.enable_proprio_history = self.cfg['env']['enableProprioHistory']
        self.use_obs_as_prop = self.cfg["env"]["useObsAsProp"]
        self.enable_action_history = self.cfg['env']['enableActionHistory']
        self.enable_point_cloud = self.cfg["env"]["enablePointCloud"]
        self.enable_attn_mask = self.cfg["env"]["enableAttnMask"]
        self.input_priv = self.cfg["env"]["input_priv"]
        
        self.domain_rand_config = self.cfg["task"]["domain_randomization"]

        self.traj_pc = None

        self.log_video = self.cfg["env"]["enableVideoLog"]
        self.recording_enabled = False
        if self.log_video:
            self.log_cam_pos = gymapi.Vec3(*[0.50, -0.15, 0.65])
            self.log_cam_target =  gymapi.Vec3(*[0.0, -0.15, 0.6])
            self.video_frames = []
            if self.dexycb_objects:
                envs = self.cfg["env"]["numEnvs"]
                self.video_log_idx = np.arange(np.minimum(envs,len(self.cfg['env']['DexYcbObjects'])))
            else:
                self.video_log_idx = np.arange(9) #self.cfg["env"]["videoLogIdx"]
            self.logging_camera = []
            self.logging_camera_tensor = []
            self.video_log_freq = self.cfg["env"]["videoLogFreq"]
            force_render = True 

        # 1.0 means keypoints correspond to the corners of the object
        # larger values help the agent to prioritize rotation matching
        self.keypoint_scale = self.cfg["env"]["keypointScale"]
        self.point_cloud_scale = self.cfg["env"]["pointCloudScale"]


        self.with_dof_force_sensors = False
        # create fingertip force-torque sensors
        self.with_fingertip_force_sensors = False

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (self.control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time),
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]

        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")

        if self.cfg["env"]["lowmem"]:
            self.asset_files_dict = load_asset_files_ycb_lowmem(self.asset_root, folder_name=self.cfg["env"]["urdfFolder"])
        else:
            self.asset_files_dict = load_asset_files_ycb(self.asset_root,folder_name=self.cfg["env"]["urdfFolder"])


        self.bucket_asset_file = "urdf/bucket.urdf"
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

        # can be only "full_state"? 
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_state"]):
            raise Exception("Unknown type of observations!") #WHY

        print("Obs type:", self.obs_type)

        num_dof_pos = self.num_hand_arm_dofs
        num_dof_vel = self.num_hand_arm_dofs
        num_dof_forces = self.num_hand_arm_dofs if self.with_dof_force_sensors else 0

        palm_pos_size = 3
        palm_rot_vel_angvel_size = 10

        obj_vel_angvel_size = 6

        fingertip_rel_pos_size = 3 * self.num_allegro_fingertips

        camera_img_size = 0 #here camera is only used for collecting data and not training

        keypoint_info_size = self.num_keypoints * 3 + self.num_keypoints * 3
        object_scales_size = 3
        max_keypoint_dist_size = 1
        lifted_object_flag_size = 1
        progress_obs_size = 1 + 1
        closest_fingertip_distance_size = self.num_allegro_fingertips
        closest_hand_joint_pose_distance_size =  len(self.hand_joints) if self.use_hand_joint_pose_reward else 0 
        curr_hand_joint_pose_distance_size = len(self.hand_joints) if self.use_hand_joint_pose_reward else 0 
        reward_obs_size = 1
        goal_pos_size = 3

        if self.cfg["env"]["input_priv"]:
            priv_input_size = 5 
        else:
            priv_input_size = 0 

        self.full_state_size = (
            num_dof_pos
            + num_dof_vel
            + num_dof_forces
            + goal_pos_size
            + camera_img_size
            + priv_input_size
            + self.point_cloud_sampled_dim*3 #for point cloud
            + palm_pos_size
            + palm_rot_vel_angvel_size
            + obj_vel_angvel_size
            + fingertip_rel_pos_size
            + keypoint_info_size
            #+ object_scales_size
            + max_keypoint_dist_size
            + lifted_object_flag_size
            + progress_obs_size
            + closest_fingertip_distance_size
            + reward_obs_size 
            + closest_hand_joint_pose_distance_size
            + curr_hand_joint_pose_distance_size 
            # + self.num_allegro_actions
        )


        self.point_cloud_begin_index = (
            num_dof_pos
            + num_dof_vel
            + num_dof_forces
        )

        self.point_cloud_end_index = (
            num_dof_pos
            + num_dof_vel
            + num_dof_forces
            + self.point_cloud_sampled_dim*3 
        )


        self.object_asset_scales = [[1.0,1.0,1.0] for _ in range(len(list(self.asset_files_dict.keys())))]

        num_states = self.full_state_size
        
        self.num_obs_dict = {
            "full_state": self.full_state_size,
        }

        self.up_axis = "z"

        self.fingertip_obs = True

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.num_allegro_kuka_actions

        self.cfg["device_type"] = sim_device.split(":")[0]
        self.cfg["device_id"] = int(sim_device.split(":")[1])
        self.cfg["headless"] = headless

        self.goal_object_indices = []
        self.goal_asset = None


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

        if self.obs_type == "full_state":
            if self.with_fingertip_force_sensors:
                sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
                self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                    self.num_envs, self.num_allegro_fingertips * 6
                )

            if self.with_dof_force_sensors:
                dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
                self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                    self.num_envs, self.num_hand_arm_dofs
                )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

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
        #desired_kuka_pos = torch.tensor([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])  # pose v1
        #desired_kuka_pos = torch.tensor([-2.135, 0.843, 1.786, -0.903, -2.262, 1.301, -2.791])  # pose v2
        #desired_kuka_pos = torch.tensor([1.54, -0.90, 0.89, 2.01, 1.42, -0.15, 1.00])  # pose v3
        # desired_kuka_pos = torch.tensor([1.74, -0.39, -2.45, 3.03, -4.45, -0.76, 2.58])  # pose v4
        # desired_kuka_pos = -1*torch.tensor([4.06,-0.62,0.32,1.11,-0.32,-0.94,2.64]) #pose v6 
        # desired_kuka_pos = torch.tensor([4.06,-0.62,0.32,-1.11,-0.32,-0.94,0.71]) #pose v7 new init pose
        # desired_kuka_pos = torch.tensor([4.06,-0.62,0.32,1.11,-0.32,-0.94,2.64]) #pose v5 
        if self.cfg["env"]["initPoseVersion"] == "v10" or self.cfg["env"]["initPoseVersion"] == "v11" or self.cfg["env"]["initPoseVersion"] == "v12" or self.cfg["env"]["initPoseVersion"] == "v13" or self.cfg["env"]["initPoseVersion"] == "v14":
            self.hand_arm_default_dof_pos = torch.tensor(kuka_init_poses[self.cfg["env"]["initPoseVersion"]], dtype=torch.float, device=self.device)
        else:
            self.hand_arm_default_dof_pos[:7] = torch.tensor(kuka_init_poses[self.cfg["env"]["initPoseVersion"]], dtype=torch.float, device=self.device)


        #take the initial position of both the arm and the hand from the file
        if self.use_saved_init_pos: 
            import pickle as pkl 
            self.hand_joint_target_ref = pkl.load(open("target_reference.pkl", "rb"))
            self.hand_joint_target_ref = ptu.to_torch(self.hand_joint_target_ref,device=self.device)
            self.object_ref_idx = torch.tensor([self.hand_joint_target_ref['object_names'].index(name) for name in self.object_names])
            self.object_ref_idx = self.object_ref_idx.to(device=self.device)
            self.hand_arm_default_dof_pos = self.hand_joint_target_ref['tgt_ref_hnd_pose'][0]

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
        self.true_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )


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

        if self.use_hand_joint_pose_reward:
            import pickle as pkl 
            self.hand_joint_target_ref = pkl.load(open("target_reference.pkl", "rb"))
            self.hand_joint_target_ref = ptu.to_torch(self.hand_joint_target_ref,device=self.device)
            self.object_ref_idx = torch.tensor([self.hand_joint_target_ref['object_names'].index(name) for name in self.object_names])
            self.object_ref_idx = self.object_ref_idx.to(device=self.device)

    
        self.hand_forces = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.contact_forces = torch.zeros((self.num_envs, self.num_hand_arm_dofs), dtype=torch.float, device=self.device)

        reward_keys = [
            "raw_fingertip_delta_rew",
            "raw_fingertip_shape_delta_rew",
            "raw_hand_delta_penalty",
            "raw_palm_delta_rew",
            "raw_hand_joint_pose_rew",
            "raw_lifting_rew",
            "raw_keypoint_rew",
            "fingertip_delta_rew",
            "fingertip_shape_delta_rew",
            "palm_delta_rew",
            "hand_delta_penalty",
            "lifting_rew",
            "lift_bonus_rew",
            "hand_joint_pose_rew",
            "keypoint_rew",
            "bonus_rew",
            "kuka_actions_penalty",
            "allegro_actions_penalty",
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
        if self.should_load_initial_states:
            self.load_initial_states()

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

        # hand_joint_point_clouds = []
        #get hand joint point clouds 

        # joint_meshfiles = get_link_meshes_from_urdf(os.path.join(asset_root_robot,hand_arm_asset_file), self.hand_joints)

        # for joint_name,meshfile in zip(self.hand_joints,joint_meshfiles):
        #     joint_point_cloud_file = os.path.join(self.asset_root, 'urdf' , meshfile) #hard coded for now
        #     joint_point_cloud = trimesh.load(joint_point_cloud_file)
        #     joint_point_cloud = trimesh.sample.sample_surface(joint_point_cloud, self.hand_joint_point_cloud_sampled_dim, seed = 0)[0]
        #     joint_point_cloud = np.array(joint_point_cloud)
        #     hand_joint_point_clouds.append(joint_point_cloud)

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
       
        object_assets = []
        object_meshes = []
        object_names = []

        for k,v in self.asset_files_dict.items():
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            idx = k.split('_')[0]

            if self.simple_objects:
                if k.split('_')[0] not in self.cfg["env"]["simpleObjects"]:
                    continue 
            
            if self.very_simple_objects:
                if k.split('_')[0] not in self.cfg["env"]["verysimpleObjects"]:
                    continue 
            
            if self.dexycb_objects:
                if k.split('_')[0] not in self.cfg["env"]["DexYcbObjects"]:
                    continue

            if self.cfg["env"]["enableVhacd"]:
                object_asset_options.vhacd_enabled=True
                if k.split('_')[0] in self.cfg["env"]["vhacdObjects"]:
                    object_asset_options.vhacd_params.resolution = 10000000
                    object_asset_options.vhacd_params.beta = 0.001
                    object_asset_options.vhacd_params.alpha = 0.001
            

            object_asset =self.gym.load_asset(self.sim,asset_root,v['urdf'],object_asset_options)
            object_assets.append(object_asset)
            src_urdf = read_xml(os.path.join(asset_root,v['urdf']))
            scale = np.array([float(src_urdf.findall(f'.//collision/geometry/')[0].attrib['scale'].split(' ')[j]) for j in range(3)])
            object_pc_file = v['mesh']
            object_pc = trimesh.load(os.path.join(asset_root,object_pc_file))
            mesh = np.array(trimesh.sample.sample_surface(object_pc, self.point_cloud_sampled_dim, seed=0)[0])*scale #no notion of scale
            # mesh = object_pc 
            object_meshes.append(mesh)
            object_names.append(k)
            
        #assuming this to be same for all of them
        object_rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])  
        object_shape_count = self.gym.get_asset_rigid_shape_count(object_assets[0])
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shape_count

        # load auxiliary objects
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, self.table_asset_file, table_asset_options)

        print("ADDED TABLE TO SIM")
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = allegro_pose.p.x
        table_pose_dy, table_pose_dz = -0.50, 0.03  #-0.15, 0.023 
        
        if self.randomization_params["randomize_table_position"]:
            table_pose_dy = -1*np.random.uniform(self.randomization_params["table_y_lower"], 
                                              self.randomization_params["table_y_upper"])
            
            table_pose_dz = np.random.uniform(self.randomization_params["table_z_lower"], 
                                              self.randomization_params["table_z_upper"])


        table_pose.p.y = allegro_pose.p.y + table_pose_dy
        table_pose.p.z = allegro_pose.p.z + table_pose_dz


        # table_pose.p.y = self.cfg["env"]["tablePosey"] #-0.15 
        # table_pose.p.z = self.cfg["env"]["tablePosez"] #0.023

        # table_pose_dy = table_pose.p.y - allegro_pose.p.y
        # table_pose_dz = table_pose.p.z - allegro_pose.p.z

        self.table_pose = table_pose 

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        #What is this used for??
        additional_rb, additional_shapes = self._load_additional_assets(object_asset_root, allegro_pose)
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(allegro_pose, table_pose_dy, table_pose_dz)

        self.allegro_hands = []
        self.envs = []

        object_init_state = []

        self.allegro_hand_indices = []
        self.object_handles = []
        object_indices = []
        object_scales = []
        object_point_clouds = []
        object_keypoint_offsets = []
        self.rgb_tensors = []
        self.depth_tensors = []
        self.seg_tensors = []
        self.cams = []
        self.object_names = []
        self.bucket_object_indices = []


        self.allegro_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]

        self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "link7")


        #check where this is used. Probably for putting penalty on table collision. 
        self.hand_handles = [self.gym.find_asset_rigid_body_index(allegro_kuka_asset, f"link{i}") for i in range(2,7)]
        self.allegro_handles = [self.gym.find_asset_rigid_body_index(allegro_kuka_asset, f"link_{i}.0") for i in range(15)]

        # this rely on the fact that objects are added right after the arms in terms of create_actor()
        self.object_rb_handles = list(range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count))


        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies*25, max_agg_shapes*25, True)

            allegro_actor = self.gym.create_actor(env_ptr, allegro_kuka_asset, allegro_pose, "allegro", i, -1, 2)

            populate_dof_properties(allegro_hand_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_hand_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.allegro_hand_indices.append(allegro_hand_idx)

            if self.obs_type == "full_state":
                if self.with_fingertip_force_sensors:
                    for ft_handle in self.allegro_fingertip_handles:
                        env_sensors = [self.gym.create_force_sensor(env_ptr, ft_handle, allegro_sensor_pose)]
                        self.allegro_sensors.append(env_sensors)

                if self.with_dof_force_sensors:
                    self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_actor)
            
            
            # if self.enable_camera_sensors:
            #     self.k_frame_saved = 0 #for debugging purposes

            #     #cameras around the table 
            #     self.num_cameras = 1
            #     for j in range(self.num_cameras):
            #         cam_props = gymapi.CameraProperties()
            #         cam_props.width = self.cam_w 
            #         cam_props.height = self.cam_h 
            #         cam_props.horizontal_fov = self.cam_fov
            #         cam_props.supersampling_horizontal = self.cam_ss 
            #         cam_props.supersampling_vertical = self.cam_ss 
            #         cam_props.enable_tensors = True

            #         cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
            #         self.gym.set_camera_location(cam_handle, env_ptr, self.cam_pos[j], self.cam_target[j])
            #         self.cams.append(cam_handle)
            #         cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
            #         cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
            #         self.rgb_tensors.append(cam_tensor_th)

            #         depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_DEPTH)
            #         depth_tensor_th = gymtorch.wrap_tensor(depth_tensor)
            #         self.depth_tensors.append(depth_tensor_th)

            #         seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_SEGMENTATION)
            #         seg_tensor_th = gymtorch.wrap_tensor(seg_tensor)
            #         self.seg_tensors.append(seg_tensor_th)      

            #     self.enable_wrist_camera = False
            #     if self.enable_wrist_camera:
            #     # Camera
            #     #Intrinsic Values of wrist camera right now 
            #     #are not matched to the one outside rn
            #         self.k_frame_saved = 0
            #         cam_props = gymapi.CameraProperties()
            #         cam_props.width = self.cam_w 
            #         cam_props.height = self.cam_h 
            #         cam_props.horizontal_fov = self.cam_fov
            #         cam_props.supersampling_horizontal = self.cam_ss 
            #         cam_props.supersampling_vertical = self.cam_ss 
            #         cam_props.enable_tensors = True
            #         cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
            #         rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link7")
            #         local_t = gymapi.Transform()
            #         local_t.p = gymapi.Vec3(*self.cam_loc_p)
            #         xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
            #         local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
            #         self.gym.attach_camera_to_body(cam_handle, env_ptr, rigid_body_hand_ind, local_t, gymapi.FOLLOW_TRANSFORM)
            #         self.cams.append(cam_handle)
            #         # Camera tensor
            #         rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
            #         cam_tensor_th = gymtorch.wrap_tensor(rgb_tensor)
            #         self.rgb_tensors.append(cam_tensor_th)


            #         depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_DEPTH)
            #         depth_tensor_th = gymtorch.wrap_tensor(depth_tensor)
            #         self.depth_tensors.append(depth_tensor_th)
                    
            #         seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_SEGMENTATION)
            #         seg_tensor_th = gymtorch.wrap_tensor(seg_tensor)
            #         self.seg_tensors.append(seg_tensor_th)    

            # #if video logging is on
            # if self.log_video and i in self.video_log_idx:
            #     cam_props = gymapi.CameraProperties()
            #     cam_props.width = self.cam_w 
            #     cam_props.height = self.cam_h 
            #     cam_props.horizontal_fov = self.cam_fov
            #     cam_props.supersampling_horizontal = self.cam_ss 
            #     cam_props.supersampling_vertical = self.cam_ss 
            #     cam_props.enable_tensors = True
            #     cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
            #     self.gym.set_camera_location(cam_handle, env_ptr, self.log_cam_pos, self.log_cam_target)
            #     cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
            #     self.logging_camera.append(cam_handle)
            #     self.logging_camera_tensor.append(gymtorch.wrap_tensor(cam_tensor))

            # add object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]
            object_pc = object_meshes[object_asset_idx]
            object_point_clouds.append(object_pc)


            object_handle = self.gym.create_actor(env_ptr, object_asset, self.object_start_pose, "object", i, 0, 3)
            self.object_handles.append(object_handle)
            self.object_names.append(object_names[object_asset_idx])
            
            object_init_state.append(
                [
                    self.object_start_pose.p.x,
                    self.object_start_pose.p.y,
                    self.object_start_pose.p.z,
                    self.object_start_pose.r.x,
                    self.object_start_pose.r.y,
                    self.object_start_pose.r.z,
                    self.object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            #object_idx is idx in the gym environment not in the in-memory list
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= object_scale[coord_idx]* self.keypoint_scale / 2
                object_offsets.append(keypoint)

            object_keypoint_offsets.append(object_offsets)

            # table object
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 1)
            table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)

            bucket_handle = self.gym.create_actor(
            env_ptr, self.bucket_asset, self.bucket_pose, "bucket_object", i, 0, 4)
            bucket_object_idx = self.gym.get_actor_index(env_ptr, bucket_handle, gymapi.DOMAIN_SIM)
            self.bucket_object_indices.append(bucket_object_idx)

            if self.enable_camera_sensors:
                self.k_frame_saved = 0 #for debugging purposes

                #cameras around the table 
                self.num_cameras = 1
                for j in range(self.num_cameras):
                    cam_props = gymapi.CameraProperties()
                    cam_props.width = self.camera_config["width"] #672 #self.cam_w 
                    cam_props.height = self.camera_config["height"] #376 #self.cam_h 
                    cam_props.horizontal_fov = 180 * 2* math.atan(cam_props.width/(2*self.camera_config["fx"]))/np.pi  #103.55 #self.cam_fov
                    
                    cam_props.enable_tensors = True

                    cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                    table_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr,table_handle, "box")

                    local_t = gymapi.Transform()
                    self.cam_loc_p = np.array(self.camera_config["pose"]) 
                    self.cam_loc_p += self.cam_pose_rnd*np.random.uniform(-1,1,3)
                    local_t.p = gymapi.Vec3(*list(self.cam_loc_p)) - table_pose.p
                    Rot = np.array(self.camera_config["R"])
                    sapien_frame_conversion = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
                    final_rot = Rot @ sapien_frame_conversion
                    q_xyzw= R.from_matrix(final_rot).as_quat()
                    #quat -> euler -> euler -> quat  for adding randomization
                    # euler = R.from_quat(q_xyzw).as_euler('zyx')
                    # euler += np.random.uniform(-np.pi*self.cam_rot_rnd/180, np.pi*self.cam_rot_rnd/180, 3)
                    # q_xyzw = R.from_euler('zyx', euler).as_quat()
                    self.cam_rnd_R = torch.eye(3).repeat(self.num_envs, 1, 1).to(self.device)

                    local_t.r = gymapi.Quat(*q_xyzw)

                    self.gym.attach_camera_to_body(cam_handle, env_ptr, table_hand_ind, local_t, gymapi.FOLLOW_TRANSFORM)
                    #self.gym.set_camera_location(cam_handle, env_ptr, self.cam_pos[j], self.cam_target[j])

                    self.cams.append(cam_handle)
                    cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                    cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                    self.rgb_tensors.append(cam_tensor_th)

                    depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_DEPTH)
                    depth_tensor_th = gymtorch.wrap_tensor(depth_tensor)
                    self.depth_tensors.append(depth_tensor_th)

                    seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    seg_tensor_th = gymtorch.wrap_tensor(seg_tensor)
                    self.seg_tensors.append(seg_tensor_th)      

                self.enable_wrist_camera = False
                if self.enable_wrist_camera:
                # Camera
                #Intrinsic Values of wrist camera right now 
                #are not matched to the one outside rn
                    self.k_frame_saved = 0
                    cam_props = gymapi.CameraProperties()
                    cam_props.width = self.cam_w 
                    cam_props.height = self.cam_h 
                    cam_props.horizontal_fov = self.cam_fov
                    cam_props.supersampling_horizontal = self.cam_ss 
                    cam_props.supersampling_vertical = self.cam_ss 
                    cam_props.enable_tensors = True
                    cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                    rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, "link7")
                    local_t = gymapi.Transform()
                    local_t.p = gymapi.Vec3(*self.cam_loc_p)
                    xyz_angle_rad = [np.radians(a) for a in self.cam_loc_r]
                    local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                    self.gym.attach_camera_to_body(cam_handle, env_ptr, rigid_body_hand_ind, local_t, gymapi.FOLLOW_TRANSFORM)
                    self.cams.append(cam_handle)
                    # Camera tensor
                    rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_COLOR)
                    cam_tensor_th = gymtorch.wrap_tensor(rgb_tensor)
                    self.rgb_tensors.append(cam_tensor_th)


                    depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_DEPTH)
                    depth_tensor_th = gymtorch.wrap_tensor(depth_tensor)
                    self.depth_tensors.append(depth_tensor_th)
                    
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    seg_tensor_th = gymtorch.wrap_tensor(seg_tensor)
                    self.seg_tensors.append(seg_tensor_th)    


            # task-specific objects (i.e. goal object for reorientation task)
            # self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=object_asset_idx)
            #rn bad engineering 
            
            if self.domain_rand_config["randomize_friction"]:
                rand_friction = np.random.uniform(self.domain_rand_config["friction_lower_limit"], 
                                                  self.domain_rand_config["friction_upper_limit"])
                obj_restitution = np.random.uniform(0, 1)

                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_actor)
                for p in hand_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, allegro_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)



            if self.domain_rand_config["randomize_object_mass"]:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(self.domain_rand_config["mass_lower_limit"], 
                                               self.domain_rand_config["mass_upper_limit"])
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)


            if self.domain_rand_config["randomize_object_com"]:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            

            if self.domain_rand_config["randomize_table_friction"]:
                rand_friction = np.random.uniform(self.domain_rand_config["table_friction_lower_limit"], 
                                                  self.domain_rand_config["table_friction_upper_limit"])

                table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
                for p in table_props:
                    p.friction = rand_friction

                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_props)

            #randomization
            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_actor)

            self.gym.end_aggregate(env_ptr)


        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )

        self.goal_states = torch.zeros_like(self.object_init_state)

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.link_handles = to_torch(self.link_handles, dtype=torch.long, device=self.device)
        self.hand_joint_handles = to_torch(self.hand_joint_handles, dtype=torch.long, device=self.device)
        self.hand_handles = to_torch(self.hand_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        # self.hand_joint_point_clouds = to_torch(hand_joint_point_clouds, dtype=torch.float, device=self.device)
        self.allegro_hand_indices = to_torch(self.allegro_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)

        self.object_point_clouds = to_torch(object_point_clouds, dtype=torch.float, device=self.device)
        self.bucket_object_indices = to_torch(self.bucket_object_indices, dtype=torch.long, device=self.device)

        # self._after_envs_created()
        # try:
        #     # by this point we don't need the temporary folder for procedurally generated assets
        #     tmp_assets_dir.cleanup()
        # except Exception:
        #     pass  
    

    def get_canonical_pc(self,pc,ref_rot,ref_pos):

        if pc.dim() == 4:
            ref_rot = ref_rot.reshape(ref_rot.size(0),1,1,ref_rot.size(-1))
            ref_pos = ref_pos.reshape(ref_pos.size(0),1,1,ref_pos.size(-1))
            pc = pc - ref_pos
            ref_rot = ref_rot.repeat(1,pc.size(1),pc.size(2),1)

        elif pc.dim() == 3:
            ref_rot = ref_rot.reshape(ref_rot.size(0),1,ref_rot.size(-1))
            ref_pos = ref_pos.reshape(ref_pos.size(0),1,ref_pos.size(-1))
            pc = pc - ref_pos
            ref_rot = ref_rot.repeat(1,pc.size(1),1)
        else:
            pc = pc - ref_pos 
            ref_rot  = ref_rot

        return quat_apply_inverse(ref_rot,pc)

    def frame_transformation(self,pc,ref_rot,ref_pos):

        if pc.dim() == 4:
            ref_rot = ref_rot.reshape(ref_rot.size(0),1,1,ref_rot.size(-1))
            ref_pos = ref_pos.reshape(ref_pos.size(0),1,1,ref_pos.size(-1))
            ref_rot = ref_rot.repeat(1,pc.size(1),pc.size(2),1)
        elif pc.dim() == 3:
            ref_rot = ref_rot.reshape(ref_rot.size(0),1,ref_rot.size(-1))
            ref_pos = ref_pos.reshape(ref_pos.size(0),1,ref_pos.size(-1))
            ref_rot = ref_rot.repeat(1,pc.size(1),1)
        
        else:
            ref_rot  = ref_rot

        z = quat_apply(ref_rot,pc)
        
        return z + ref_pos

    def _hand_joint_pose_distance(self):
        """Reward for the hand joint pose.
        This reward is used to guide the hand to the desired pose. 
        Get self.hand_joint_target_point_cloud from reference"""
        #reorient the object axis 
        obj_tgt_rot = self.hand_joint_target_ref['tgt_ref_obj_rot'][self.object_ref_idx]
        obj_tgt_pos = self.hand_joint_target_ref['tgt_ref_obj_pos'][self.object_ref_idx]
        hnd_tgt_pos = self.hand_joint_target_ref['tgt_ref_hnd_pos'][self.object_ref_idx]
        obj_tgt_pc = self.hand_joint_target_ref['tgt_ref_obj_pc'][self.object_ref_idx]

        """
        hand in the target data, obj in the trget data
        hnd_tgt_pos is the cooridnate of the hand tgt pose wrt obj target 
        1 ,
        """
        hnd_tgt_pos = self.get_canonical_pc(hnd_tgt_pos,obj_tgt_rot,obj_tgt_pos)
        tgt_hnd_pos_env_frame = self.frame_transformation(hnd_tgt_pos,self.object_rot,self.object_pos)

        hnd_rel_pos = self.get_canonical_pc(self.hand_joint_pos,self.object_rot,self.object_pos)


        # TODO: Change to other type of norms
        hand_joint_pose_dist= torch.norm(hnd_tgt_pos - hnd_rel_pos, dim=-1)

        return hand_joint_pose_dist, tgt_hnd_pos_env_frame


    def _shape_distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        fingertip_shape_deltas_closest = self.closest_fingertip_shape_dist - self.curr_fingertip_shape_distances

        self.closest_fingertip_shape_dist = torch.minimum(self.closest_fingertip_shape_dist, self.curr_fingertip_shape_distances)

        fingertip_shape_deltas = torch.clip(fingertip_shape_deltas_closest, 0, 10)
        fingertip_shape_deltas *= self.finger_rew_coeffs
        fingertip_shape_delta_rew = torch.sum(fingertip_shape_deltas, dim=-1)
        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        fingertip_shape_delta_rew *= ~lifted_object

        return fingertip_shape_delta_rew

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        # update the values if finger tips got closer to the object
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)

        # again, positive is closer, negative is further away
        # here we use index of the 1st finger, when the distance is large it doesn't matter which one we use
        hand_deltas_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        # update the values if finger tips got further away from the object
        self.furthest_hand_dist = torch.maximum(self.furthest_hand_dist, self.curr_fingertip_distances[:, 0])

        # clip between zero and +inf to turn deltas into rewards
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_deltas *= self.finger_rew_coeffs
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        fingertip_delta_rew *= ~lifted_object

        # clip between zero and -inf to turn deltas into penalties
        hand_delta_penalty = torch.clip(hand_deltas_furthest, -10, 0)
        hand_delta_penalty *= ~lifted_object
        # multiply by the number of fingers so two rewards are on the same scale
        hand_delta_penalty *= self.num_allegro_fingertips

        return fingertip_delta_rew, hand_delta_penalty

    def _hand_joint_pose_reward(self,lifted_object):
        """This reward is an alternative to the distance delta reward
        Since joint pose can be maintained after lift too. This reward can be given 
        beyond lift"""
        self.closest_joint_pose_dist = self.closest_joint_tgt_rel_pose_dist - self.curr_joint_tgt_rel_pose_dist
        self.closest_joint_tgt_rel_pose_dist = torch.minimum(self.closest_joint_tgt_rel_pose_dist, self.curr_joint_tgt_rel_pose_dist)
        tgt_pose_delta = torch.clip(self.closest_joint_pose_dist, 0, 10)
        # tgt_pose_delta *= self.finger_rew_coeffs
        tgt_pose_delta_rew = torch.sum(tgt_pose_delta, dim=-1)
        
        if self.use_hand_joint_pose_reward and self.use_pose_reward_unlifted:
            tgt_pose_delta_rew *= ~lifted_object
        
        if self.use_hand_joint_pose_reward and self.use_pose_reward_lifted:
            tgt_pose_delta_rew *= lifted_object

        return tgt_pose_delta_rew

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.02 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: Tensor) -> Tensor:
        # this is positive if we got closer, negative if we're further away
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist

        # update the values if we got closer to the target
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)

        # clip between zero and +inf to turn deltas into rewards
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)

        # administer reward only when we already lifted an object from the table
        # to prevent the situation where the agent just rolls it around the table
        keypoint_rew = max_keypoint_deltas * lifted_object

        return keypoint_rew

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

        # resets = torch.where(self.object_pos[:, 0] < -1*0.3 , torch.ones_like(self.reset_buf), self.reset_buf)  # fall
        # resets =  torch.where(self.object_pos[:, 0] > 0.45 , torch.ones_like(resets), resets)  # fall
        # resets = torch.where(torch.abs(self.object_pos[:, 1] + 0.05) > 0.22, torch.ones_like(resets), resets)  # fall

        resets = torch.where(self.object_pos[:, 2] < 0.06, torch.ones_like(self.reset_buf), self.reset_buf)  # fall
        
        if self.max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            # self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= self.max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(resets), resets)
        resets = self._extra_reset_rules(resets)
        return resets

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective

    # def compute_pixel_obs(self):
    #     self.gym.render_all_camera_sensors(self.sim)
    #     self.gym.start_access_image_tensors(self.sim)
    #     # crop_l = (self.cam_w - self.i) //2
    #     # crop_r = crop_l + self.im_size
    #     for i in range(self.num_envs):
    #         if i == 0:
    #             #check if debug_frames exists and make it if not 
    #             if not os.path.exists(os.path.join(os.getcwd(), "debug_frames")):
    #                 os.mkdir(os.path.join(os.getcwd(), "debug_frames"))
    #             frame_name = os.path.join(os.getcwd(), "debug_frames", "frame_{:05d}.jpg".format(self.k_frame_saved))
    #             plt.imsave(frame_name, self.cam_tensors[2*i].cpu().numpy())
    #             self.k_frame_saved += 1
            
    #         # img = self.cam_tensors[i][:, crop_l:crop_r, :3].float()
    #         self.frame_buf[i] = self.cam_tensors[2*i][:, :, :3].float() #take first camera for frame_buf 
            
    #     self.gym.end_access_image_tensors(self.sim)
    #     return

    def capture_image(self):
        #does not capture depth
        #currently implemented only for one camera
        # images = [] 
        # for i in range(self.num_cameras):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        image = torch.stack(self.rgb_tensors).to(self.device)
        # simulate sensor noise
        # image = self.rgbd_center_crop(image.permute(0, 3, 1, 2))
        image = image.permute((0,3,1,2))
        image = image.float()
        self.gym.end_access_image_tensors(self.sim)
            # images.append(image)
        return image

    def capture_logging_image(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        all_images = []
        num_cameras = len(self.logging_camera_tensor)
        for i in range(num_cameras):
            image = self.logging_camera_tensor[i].to(self.device)
            image = image[:,:,:3]
            image = image.int()
            all_images.append(image)
        self.gym.end_access_image_tensors(self.sim)

        num_rows = np.minimum(num_cameras, 3)
        num_cols = np.ceil(num_cameras/num_rows).astype(int)

        combined_image = torch.zeros((self.cam_h*num_rows,self.cam_w*num_cols,3),dtype=torch.int)
        for i in range(num_cameras):
            row = i//num_cols
            col = i%num_cols
            combined_image[row*self.cam_h:(row+1)*self.cam_h,col*self.cam_w:(col+1)*self.cam_w,:] = all_images[i]

        return combined_image
    
    def _palm_delta_reward(self, lifted_object: Tensor) -> Tensor:

        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        palm_deltas_closest = self.closest_palm_dist - self.curr_palm_distances
        # update the values if finger tips got closer to the object
        self.closest_palm_dist = torch.minimum(self.closest_palm_dist, self.curr_palm_distances)

        # clip between zero and +inf to turn deltas into rewards
        palm_deltas = torch.clip(palm_deltas_closest, 0, 10)

        palm_delta_rew = torch.sum(palm_deltas, dim=-1)
        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        palm_delta_rew *= ~lifted_object

        palm_delta_rew *= self.num_allegro_fingertips
        
        return palm_delta_rew


    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:

        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        palm_delta_rew = self._palm_delta_reward(lifted_object)
        fingertip_shape_delta_rew = self._shape_distance_delta_rewards(lifted_object) 
        hand_joint_pose_rew= self._hand_joint_pose_reward(lifted_object)

        keypoint_rew = self._keypoint_reward(lifted_object)

        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        # noinspection PyTypeChecker
        self.near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance

        #bugfix begin 25/12/23
        self.near_goal_steps = torch.where(
            self.near_goal, self.near_goal_steps + 1, torch.zeros_like(self.near_goal_steps))
        # self.near_goal_steps += self.near_goal
        #bugfix end 

        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = torch.zeros_like(is_success) #is_success another thing! goal_reset happens on each success
        self.is_success = is_success 
        self.successes += is_success

        self.reset_goal_buf[:] = goal_resets

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_fingertip_shape_delta_rew"] += fingertip_shape_delta_rew
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew
        self.rewards_episode["raw_palm_delta_rew"] += palm_delta_rew
        self.rewards_episode["raw_hand_joint_pose_rew"] += hand_joint_pose_rew


        
        # Reaching
        if not self.use_fingertip_shape_distance_reward:
            fingertip_shape_delta_rew *= 0 

        if not self.use_fingertip_reward:
            fingertip_delta_rew *= 0
        
        if not self.use_palm_reward:
            palm_delta_rew *= 0
        
        if not self.use_hand_joint_pose_reward:
            hand_joint_pose_rew *= 0
        
        # Grasping
        if not self.use_lifting_reward:
            lifting_rew *= 0 
        
        if not self.use_keypoint_reward:
            keypoint_rew *= 0

        hand_delta_penalty *= 0

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        fingertip_shape_delta_rew *= self.distance_delta_rew_scale
        hand_joint_pose_rew *= self.distance_delta_rew_scale 


        palm_delta_rew *= self.distance_delta_rew_scale
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale

        kuka_actions_penalty, allegro_actions_penalty = self._action_penalties()

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        # We spread out the reward over "success_steps"
        bonus_rew = self.near_goal * (self.reach_goal_bonus / self.success_steps)

        reward = (
            fingertip_delta_rew
            + fingertip_shape_delta_rew
            + palm_delta_rew
            + hand_delta_penalty  # + sign here because hand_delta_penalty is negative
            + hand_joint_pose_rew 
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + allegro_actions_penalty
            + bonus_rew
        )

        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        self.extras["successes"] = self.prev_episode_successes.mean()

        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        # scalars for logging
        self.extras["true_objective_mean"] = self.true_objective.mean()
        self.extras["true_objective_min"] = self.true_objective.min()
        self.extras["true_objective_max"] = self.true_objective.max()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (fingertip_shape_delta_rew, "fingertip_shape_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (palm_delta_rew, "palm_delta_rew"),
            (hand_joint_pose_rew, "hand_joint_pose_rew"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (allegro_actions_penalty, "allegro_actions_penalty"),
            (bonus_rew, "bonus_rew"),
        ]

        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
        self.extras["rewards_episode"] = self.rewards_episode

        return self.rew_buf, is_success

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
                print(f"Last ep true objective: {self.prev_episode_true_objective.mean().item():.2f}")

                self.eval_summaries.add_scalar("last_ep_successes", self.prev_episode_successes.mean().item(), frame)
                self.eval_summaries.add_scalar(
                    "last_ep_true_objective", self.prev_episode_true_objective.mean().item(), frame
                )
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

        if self.obs_type == "full_state":
            if self.with_fingertip_force_sensors:
                self.gym.refresh_force_sensor_tensor(self.sim)
            if self.with_dof_force_sensors:
                self.gym.refresh_dof_force_tensor(self.sim)

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.point_cloud_buf = quat_apply(
                self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1), self.object_point_clouds
            ) + self.object_pos[:, None]
        
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.palm_center_offset = torch.from_numpy(self.palm_offset).to(self.device).repeat((self.num_envs, 1))
        self._palm_state = self.rigid_body_states[:, self.allegro_palm_handle][:, 0:13]
        self._palm_pos = self.rigid_body_states[:, self.allegro_palm_handle][:, 0:3]
        self._palm_rot = self.rigid_body_states[:, self.allegro_palm_handle][:, 3:7]
        self.palm_center_pos = self._palm_pos + quat_rotate(self._palm_rot, self.palm_center_offset)

        self.fingertip_state = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:3]
        self.fingertip_rot = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 3:7]

        self.link_pos = self.rigid_body_states[:, self.link_handles][:, :, 0:3]
        self.finger_pos = self.rigid_body_states[:, self.allegro_handles][:,:,:3]

        self.hand_joint_state = self.rigid_body_states[:, self.hand_joint_handles][:, :, 0:13]
        self.hand_joint_pos = self.rigid_body_states[:, self.hand_joint_handles][:, :, 0:3]
        self.hand_joint_rot = self.rigid_body_states[:, self.hand_joint_handles][:, :, 3:7]

        # self.hand_joint_point_cloud_buf = quat_apply(
        #     self.hand_joint_rot[:, :, None].repeat(1, 1, self.point_cloud_sampled_dim, 1),
        #     self.hand_joint_point_clouds[None,:,:].repeat(self.num_envs, 1, 1, 1),
        # ) + self.hand_joint_pos[:, :, None]



        self.hand_forces = torch.norm(self.rigid_body_force_tensor[:,self.hand_handles],dim=(-1,-2))

        self.contact_forces = torch.norm(self.rigid_body_force_tensor[:,:self.num_hand_arm_bodies],dim=-2)

        if not isinstance(self.fingertip_offsets, torch.Tensor):
            self.fingertip_offsets = (
                torch.from_numpy(self.fingertip_offsets).to(self.device).repeat((self.num_envs, 1, 1))
            )

        if hasattr(self, "fingertip_pos_rel_object"):
            self.fingertip_pos_rel_object_prev[:, :, :] = self.fingertip_pos_rel_object
        else:
            self.fingertip_pos_rel_object_prev = None

        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for i in range(self.num_allegro_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + quat_rotate(
                self.fingertip_rot[:, i], self.fingertip_offsets[:, i]
            )

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_allegro_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)

        self.curr_palm_distances = torch.norm(self.palm_center_pos - self.object_pos, dim=-1).unsqueeze(-1)


        pc_num = self.object_point_clouds.shape[1]
        self.curr_fingertip_shape_distances = torch.norm(
            self.fingertip_pos_offset.unsqueeze(2).repeat(1,1,pc_num,1) 
            - self.point_cloud_buf.unsqueeze(1).repeat(1,self.num_allegro_fingertips,1,1), dim=-1
        )

        self.curr_fingertip_shape_distances = torch.min(self.curr_fingertip_shape_distances,dim=-1).values 

        if not self.use_hand_joint_pose_reward:
            self.curr_joint_tgt_rel_pose_dist = torch.zeros_like(self.closest_joint_tgt_rel_pose_dist)
        else:
            self.curr_joint_tgt_rel_pose_dist, self.hnd_tgt = self._hand_joint_pose_distance()

        # when episode ends or target changes we reset this to -1, this will initialize it to the actual distance on the 1st frame of the episode
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )

        self.closest_fingertip_shape_dist = torch.where(
            self.closest_fingertip_shape_dist < 0.0, self.curr_fingertip_shape_distances, self.closest_fingertip_shape_dist
        )


        self.closest_palm_dist = torch.where(
            self.closest_palm_dist < 0.0, self.curr_palm_distances, self.closest_palm_dist
        )

        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )

        self.closest_joint_tgt_rel_pose_dist = torch.where(
            self.closest_joint_tgt_rel_pose_dist < 0.0, self.curr_joint_tgt_rel_pose_dist, self.closest_joint_tgt_rel_pose_dist
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(1, self.num_allegro_fingertips, 1)
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos
            self.goal_keypoint_pos[:, i] = self.goal_pos

        # if self.use_fixed_height_as_target:
        #     # Only use the height as target
        #     # Fix the obj_keypoint_pos
        #     self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        #     # Zero the first two dims
        #     self.keypoints_rel_goal[:, :, 0:2] = 0
        #     self.keypoint_distances_l2 = (self.obj_keypoint_pos[:, :, 2] - self.goal_keypoint_pos[:, :, 2]).abs()
        #     # There is only a single keypoint, so we can just take the first one
        #     self.keypoints_max_dist = self.keypoint_distances_l2[:, 0]
        # else:
        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        # furthest keypoint from the goal
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values


        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(1, self.num_keypoints, 1)
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_center_repeat

        # this is the closest the keypoint had been to the target in the current episode (for the furthest keypoint of all)
        # make sure we initialize this value before using it for obs or rewards
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )
        return 


    def randomize_depth(self, depth, seg):

        num_wires = 2 
        
        # def create_mask(tensor, point1, point2):

        #     # Extract the coordinates of the points
        #     x1, y1 = point1
        #     x2, y2 = point2
            
        #     # Calculate the slope (m) and intercept (c) of the line y = mx + c
        #     if x2 != x1:
        #         m = (y2 - y1) / (x2 - x1)
        #         c = y1 - m * x1
        #     else:
        #         m = float('inf')
        #         c = None
            
        #     # Create a grid of coordinates
        #     height, width = tensor.shape[-2], tensor.shape[-1]
        #     x_coords = torch.arange(width).view(1, -1).expand(height, -1).to(tensor.device)
        #     y_coords = torch.arange(height).view(-1, 1).expand(-1, width).to(tensor.device)
            
        #     # Calculate horizontal distance from each point to the line
        #     if m != float('inf'):
        #         distance = torch.abs(y_coords - m * x_coords - c) / torch.sqrt(1 + m**2)
        #     else:
        #         distance = torch.abs(x_coords - x1)
            
        #     # Create the mask
        #     mask = distance < 2
        
        #     return mask


        # for i in range(num_wires):
        #     #choose starting point of wire at a pixel where seg is 1
        #     seg_nonzero = torch.nonzero(seg, as_tuple=False)
        #     if seg_nonzero.shape[0] == 0:
        #         start = torch.randint(0, depth.shape[-2], (2,)).to(depth.device)
        #     else:
        #         rnd_idx = torch.randint(0, seg.shape[0], (2,)).to(depth.device)
        #         start = seg_nonzero[rnd_idx][0]
        #     #choose ending point of wire somewere within 100x100 sqare centered at the starting point

            
        #     end_point = start + torch.randint(-50, 50, (2,)).to(depth.device)
        #     #make sure end point lies between depth.shape[0] and depth.shape[1]
        #     end_point[1] = torch.clamp(end_point[1], 0, depth.shape[-1]-1)
        #     end_point[0] = torch.clamp(end_point[0], 0, depth.shape[-2]-1)
        #     #draw a line between start and end point
        #     mask = create_mask(depth, start, end_point)
            
        #     mask = mask.repeat(depth.shape[0], 1, 1)


        #     depth[mask] = torch.tensor(self.cam_loc_p[1]).to(depth.device) + 0.10*torch.randn((1,)).to(depth.device)
            
        return depth

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
        


        #add randomization
        #depth =  self.randomize_depth(depth, (seg[0] ==2) )

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
            #fill proprio history buffers 
            # at_reset_env_ids = torch.where(self.reset_buf == 1)[0]
            # temp = self.arm_hand_dof_pos
            # proprio_history = temp[at_reset_env_ids]
            # proprio_history = proprio_history.unsqueeze(1).repeat(1,self.proprio_buf.shape[1],1).clone()
            # self.proprio_buf[at_reset_env_ids] = proprio_history

            # not_at_reset_env_ids = torch.where(self.reset_buf == 0)[0]
            # self.proprio_buf[not_at_reset_env_ids, :] = torch.cat([
            #     self.proprio_buf[not_at_reset_env_ids, 1:],
            #     temp[not_at_reset_env_ids].unsqueeze(1),
            # ], dim=1)
            
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
        

        ###save stuff for logging 
        # if not os.path.exists(os.path.join(os.getcwd(), "debug/depth")):
        #     os.makedirs(os.path.join(os.getcwd(), "debug/depth"))
        
        # np.save(os.path.join(os.getcwd(), "debug/depth", "depth_{:05d}.npy".format(self.k_frame_saved)), depth[28].cpu().numpy())
        # self.k_frame_saved += 1


    def compute_full_state(self, buf: Tensor, rewards: Tensor) -> Tuple[int, int]:
        num_dofs = self.num_hand_arm_dofs
        ofs = 0

        # dof positions
        buf[:, ofs : ofs + num_dofs] = unscale(
            self.arm_hand_dof_pos[:, :num_dofs],
            self.arm_hand_dof_lower_limits[:num_dofs],
            self.arm_hand_dof_upper_limits[:num_dofs],
        )
        ofs += num_dofs

        # dof velocities
        buf[:, ofs : ofs + num_dofs] = self.arm_hand_dof_vel[:, :num_dofs]
        ofs += num_dofs

        if self.with_dof_force_sensors:
            # dof forces
            buf[:, ofs : ofs + num_dofs] = self.dof_force_tensor[:, :num_dofs]
            ofs += num_dofs


        # # Add pixel information
        # if self.enable_camera_sensors:
        #     self.compute_pixel_obs()
        
        if self.input_priv:
            buf[:,ofs:ofs+5] = self.priv_buf 
            ofs += 5 

        #add object point cloud 
        whole_pc = self.point_cloud_buf.reshape(self.num_envs, -1)
        buf[:, ofs : ofs + self.point_cloud_sampled_dim * 3] = whole_pc
        ofs += self.point_cloud_sampled_dim*3

        # goal pos
        buf[:, ofs : ofs + 3] = self.goal_pos
        ofs += 3

        # palm pos
        buf[:, ofs : ofs + 3] = self.palm_center_pos
        ofs += 3

        # palm rot, linvel, ang vel
        buf[:, ofs : ofs + 10] = self._palm_state[:, 3:13]
        ofs += 10

        # object linvel ang vel
        buf[:, ofs : ofs + 6] = self.object_state[:, 7:13]
        ofs += 6

        # fingertip pos relative to the palm of the hand
        fingertip_rel_pos_size = 3 * self.num_allegro_fingertips
        buf[:, ofs : ofs + fingertip_rel_pos_size] = self.fingertip_pos_rel_palm.reshape(
            self.num_envs, fingertip_rel_pos_size
        )
        ofs += fingertip_rel_pos_size

        # keypoint distances relative to the palm of the hand
        keypoint_rel_pos_size = 3 * self.num_keypoints
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_palm.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        # keypoint distances relative to the goal
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_goal.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        # object scales
        # buf[:, ofs : ofs + 3] = self.object_scales
        # ofs += 3

        # closest distance to the furthest keypoint, achieved so far in this episode
        buf[:, ofs : ofs + 1] = self.closest_keypoint_max_dist.unsqueeze(-1)
        ofs += 1

        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward

        buf[:, ofs : ofs + self.num_allegro_fingertips] = self.closest_fingertip_dist
        ofs += self.num_allegro_fingertips

        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        buf[:, ofs : ofs + 1] = self.lifted_object.unsqueeze(-1)
        ofs += 1
        

        # this should help the critic predict the future rewards better and anticipate the episode termination
        buf[:, ofs : ofs + 1] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1) 
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.successes + 1).unsqueeze(-1)
        ofs += 1

        # this is where we will add the reward observation
        reward_obs_ofs = ofs
        # add rewards to observations
        reward_obs_scale = 0.01
        self.obs_buf[:, reward_obs_ofs : reward_obs_ofs + 1] = rewards.unsqueeze(-1) * reward_obs_scale

        ofs += 1

        assert ofs == self.full_state_size
        return ofs, reward_obs_ofs

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


        obj_indices = self.object_indices[env_ids]

        # reset object
        rand_pos_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        self.root_state_tensor[obj_indices] = self.object_init_state[env_ids].clone()

        # indices 0..2 correspond to the object position
        self.root_state_tensor[obj_indices, 0:1] = (
            self.object_init_state[env_ids, 0:1] + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
        )
        self.root_state_tensor[obj_indices, 1:2] = (
            self.object_init_state[env_ids, 1:2] + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
        )
        self.root_state_tensor[obj_indices, 2:3] = (
            self.object_init_state[env_ids, 2:3] + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
        )
        new_object_rot = self.get_random_quat(env_ids)

        # indices 3,4,5,6 correspond to the rotation quaternion
        if self.use_random_init_rot:    
            self.root_state_tensor[obj_indices, 3:7] = new_object_rot
        else:
            self.root_state_tensor[obj_indices,3:7] = \
                torch.zeros_like(self.root_state_tensor[obj_indices, 3:7])
            
            self.root_state_tensor[obj_indices,6] += 1.0

        self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(self.root_state_tensor[obj_indices, 7:13])

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
        # if not object_indices:
        #     # nothing to set
        #     return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def reset_idx(self, env_ids: Tensor) -> None:
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

        # randomize start object poses

        #print(env_ids)
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.reset_object_pose(env_ids)

        hand_indices = self.allegro_hand_indices[env_ids].to(torch.int32)

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device)
            + torch.log(self.force_prob_range[1])
        )

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

        if self.enable_camera_sensors:
            #euler to rotation matrix 
            cam_rot = torch_rand_float(-self.cam_rot_rnd*np.pi/180, self.cam_rot_rnd*np.pi/180, (len(env_ids), 3), device=self.device)
            self.cam_rnd_R[env_ids] = euler_to_rotation_matrix_torch(cam_rot)

                


        # if self.should_load_initial_states:
        #     if len(env_ids) > self.num_initial_states:
        #         print(f"Not enough initial states to load {len(env_ids)}/{self.num_initial_states}...")
        #     else:
        #         if self.initial_state_idx + len(env_ids) > self.num_initial_states:
        #             self.initial_state_idx = 0

        #         dof_states_to_load = self.initial_dof_state_tensors[
        #             self.initial_state_idx : self.initial_state_idx + len(env_ids)
        #         ]
        #         self.dof_state.reshape([self.num_envs, -1, *self.dof_state.shape[1:]])[
        #             env_ids
        #         ] = dof_states_to_load.clone()
        #         root_state_tensors_to_load = self.initial_root_state_tensors[
        #             self.initial_state_idx : self.initial_state_idx + len(env_ids)
        #         ]
        #         cube_object_idx = self.object_indices[0]
        #         self.root_state_tensor.reshape([self.num_envs, -1, *self.root_state_tensorprev_episode_succ.shape[1:]])[
        #             env_ids, cube_object_idx
        #         ] = root_state_tensors_to_load[:, cube_object_idx].clone()

        #         self.initial_state_idx += len(env_ids)


        # state = np.load('state.npy').astype(np.float32)[0]
        # target_qpos = np.zeros((self.num_envs, self.num_hand_arm_dofs), dtype=np.float32)[0]

        # state_torch = torch.tensor(state, dtype=torch.float32).to(self.device)
        # target_qpos = torch.tensor(target_qpos, dtype=torch.float32).to(self.device)

        # allegro_pos = state_torch[:, 0]

        # print(allegro_pos)

        # self.arm_hand_dof_pos[env_ids, :] = allegro_pos

        # self.arm_hand_dof_vel[env_ids, :] = state_torch[:, 1]

        # self.prev_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos
        # self.cur_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos


        # success = self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(state_torch))
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_qpos))

        # assert success, "Failed to set dof state and target"


        success_state = self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        success_target = self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        assert success_state and success_target

        object_indices = [self.object_indices[env_ids]]
        # object_indices.extend(self._extra_object_indices(env_ids))   ##earlier required 
        self.deferred_set_actor_root_state_tensor_indexed(object_indices)

        self.progress_buf[env_ids] = 0
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
        self.true_objective[env_ids] = 0
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

        self.extras["scalars"] = dict()
        self.extras["scalars"]["success_tolerance"] = self.success_tolerance

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
            # elif self.use_residuals:
            #     targets = self.prev_targets + self.hand_dof_speed_scale * self.pretrain_dt * self.actions
            #     self.cur_targets = tensor_clamp(
            #         targets, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
            #     )
            else:
                # targets = self.prev_targets + self.hand_dof_speed_scale * self.dt * self.actions
                #unscaled for now, may change later
                self.cur_targets = scale(
                    self.actions,
                    self.arm_hand_dof_lower_limits,
                    self.arm_hand_dof_upper_limits,
                )

                # Limits the first 7 joints to a maximum of 1/120 rad change with respect to the previous targets
                if self.limit_arm_delta_target:
                    self.cur_targets[:, :7] = tensor_clamp(self.cur_targets[:, :7],
                                                           self.prev_targets[:, :7] - self.hand_dof_speed_scale*self.dt,
                                                           self.prev_targets[:, :7] + self.hand_dof_speed_scale*self.dt)

                self.cur_targets = tensor_clamp(
                    self.cur_targets, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
                ) 
                


        self.prev_targets[:, :] = self.cur_targets[:, :]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device)
                * self.object_rb_masses
                * self.force_scale
            )


            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE
            )

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

        self._extra_curriculum()
        self.frame_since_restart += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        rewards, is_success = self.compute_kuka_reward()

        #reset should go here along with a refresh 

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        self.reset_target_pose(reset_goal_env_ids)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.set_actor_root_state_tensor_indexed()

        if len(reset_goal_env_ids) > 0 or len(reset_env_ids) > 0:
            self.compute_observations()

        #resets done 

        if self.obs_type == "full_state":
            full_state_size, reward = self.compute_full_state(self.obs_buf, rewards)
            assert (
                full_state_size == self.full_state_size
            ), f"Expected full state size {self.full_state_size}, actual: {full_state_size}"

        else:
            raise ValueError("Unkown observations type!")     


        self.clamp_obs(self.obs_buf)

        self.compute_task_buffers()

        self._eval_stats(is_success)

        if self.log_video and self.recording_enabled:
            image = self.capture_logging_image()
            self.video_frames.append(image)

        if self.viewer and self.debug_viz:
            # draw axes on target object

            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            axes_geom = gymutil.AxesGeometry(0.1)

            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.003, 8, 8, sphere_pose, color=(1, 1, 0))
            sphere_geom_white = gymutil.WireframeSphereGeometry(0.03, 8, 8, sphere_pose, color=(1, 1, 1))
            sphere_geom_red = gymutil.WireframeSphereGeometry(0.1, 8, 8, sphere_pose, color=(1, 0, 0))
            sphere_geom_purple = gymutil.WireframeSphereGeometry(0.05, 8, 8, sphere_pose, color=(1, 0, 1))

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

            goal_pos = self.goal_pos.cpu().numpy()
            for i in range(self.num_envs):
                goal_transform = gymapi.Transform()
                goal_transform.p = gymapi.Vec3(*goal_pos[i])
                gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], goal_transform)


            limit_z = self.table_pose.p.z
            for i in range(self.num_envs):
                goal_transform = gymapi.Transform()
                goal_transform.p = gymapi.Vec3(0,0,limit_z)
                gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], goal_transform)


            for i in range(self.num_envs):
                origin_transform = gymapi.Transform()
                origin_transform.p = gymapi.Vec3(0, 0, 0)
                gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[i], origin_transform)

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

            if self.traj_pc is not None:
                for j in range(self.point_cloud_sampled_dim):
                    # point_cloud_pos_cpu = self.point_cloud_buf[:,j].cpu().numpy()
                    point_cloud_pos_cpu = self.traj_pc[:,j].cpu().numpy()
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
        x_pos = torch_rand_float(self.bucket_pose.p.x, self.bucket_pose.p.x, (len(env_ids), 1), device=self.device)
        y_pos = torch_rand_float(self.bucket_pose.p.y, self.bucket_pose.p.y, (len(env_ids), 1), device=self.device)
        z_pos = torch_rand_float(self.bucket_pose.p.z, self.bucket_pose.p.z, (len(env_ids), 1), device=self.device)
        self.root_state_tensor[self.bucket_object_indices[env_ids], 0:1] = x_pos
        self.root_state_tensor[self.bucket_object_indices[env_ids], 1:2] = y_pos
        self.root_state_tensor[self.bucket_object_indices[env_ids], 2:3] = z_pos
        self.goal_states[env_ids, 0:1] = x_pos
        self.goal_states[env_ids, 1:2] = y_pos
        self.goal_states[env_ids, 2:3] = z_pos + 0.10

        # we also reset the object to its initial position
        self.reset_object_pose(env_ids)

        # since we put the object back on the table, also reset the lifting reward
        self.lifted_object[env_ids] = False

        object_indices_to_reset = [self.bucket_object_indices[env_ids], self.object_indices[env_ids]]
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
        bucket_asset_options = gymapi.AssetOptions()
        bucket_asset_options.disable_gravity = False
        bucket_asset_options.fix_base_link = True
        bucket_asset_options.collapse_fixed_joints = True
        bucket_asset_options.vhacd_enabled = True
        bucket_asset_options.vhacd_params = gymapi.VhacdParams()
        bucket_asset_options.vhacd_params.resolution = 500000
        bucket_asset_options.vhacd_params.max_num_vertices_per_ch = 32
        bucket_asset_options.vhacd_params.min_volume_per_ch = 0.001
        self.bucket_asset = self.gym.load_asset(
            self.sim, object_asset_root, self.bucket_asset_file, bucket_asset_options
        )

        self.bucket_pose = gymapi.Transform()
        self.bucket_pose.p = gymapi.Vec3()
        if self.throw_far:
            self.bucket_pose.p.x = 0.33 #arm_pose.p.x - 0.6
            self.bucket_pose.p.y = -1*0.50 #arm_pose.p.y - 1
            self.bucket_pose.p.z = 0.50 ##arm_pose.p.z + 0.45
        elif self.bucket_in_front:
            self.bucket_pose.p.x = 0.00
            self.bucket_pose.p.y = -0.35
            self.bucket_pose.p.z = 0.10   
        else:
            self.bucket_pose.p.x = 0.37
            self.bucket_pose.p.y = 0.00
            self.bucket_pose.p.z = 0.10
         
        bucket_rb_count = self.gym.get_asset_rigid_body_count(self.bucket_asset)
        bucket_shapes_count = self.gym.get_asset_rigid_shape_count(self.bucket_asset)
        print(f"Bucket rb {bucket_rb_count}, shapes {bucket_shapes_count}")

        return bucket_rb_count, bucket_shapes_count

    
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

            if obj_pc is not None:
                self.traj_pc = obj_pc

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

            # compute observatins, rewards, resets, ...
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
            
            #debug end 

            # #debug debug debug begin 
            # x = torch.cat((self.obs_buf[:,:self.point_cloud_begin_index],self.obs_buf[:,self.point_cloud_end_index:]),dim=1)
            # # self.obs_dict["proprio_"] = self.obs_buf[:,:self.raw_robot_obs_size].to(self.rl_device)
            # self.obs_dict["proprio_"] = x.to(self.rl_device)
            # #debug debug debug end 


            # #new debug for one moment 14/10/2023:1620
            # self.obs_dict["obs"][:,self.point_cloud_begin_index:self.point_cloud_end_index] = torch.zeros_like(self.obs_dict["obs"][:,self.point_cloud_begin_index:self.point_cloud_end_index])
            # #new debug end 

            # asymmetric actor-critic
            
            self.add_randomization_stats()
            
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

            return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def add_randomization_stats(self):
        
        idx = np.random.randint(0, self.num_envs)
        prop = self.gym.get_actor_rigid_body_properties(self.envs[idx],self.object_handles[idx])
        self.extras["randomization/mass_sample"] = prop[0].mass 
        prop = self.gym.get_actor_rigid_shape_properties(self.envs[idx],self.allegro_hands[idx])
        self.extras["randomization/friction_sample"] = prop[0].friction 

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
    

    def apply_init_domain_randomization(self, dr_config,env_ids):

        for i in env_ids:

            if dr_config['randomize_friction']:
                
                hand_friction, hand_restitution = randomize_friction(self.gym, self.envs[i], self.allegro_hands[i],dr_config['friction_config'])
                if not self.cfg["env"]["addZerosInPrivBuf"]:
                    self.priv_buf[i,0] = torch.tensor(hand_friction[0],device=self.rl_device,dtype=torch.float)
                    self.priv_buf[i,1] = torch.tensor(hand_restitution[0],device=self.rl_device,dtype=torch.float)

                obj_friction, obj_restitution = randomize_friction(self.gym, self.envs[i], self.object_handles[i],dr_config['friction_config'])

                if not self.cfg["env"]["addZerosInPrivBuf"]:
                    self.priv_buf[i,2] = torch.tensor(obj_friction[0],device=self.rl_device,dtype=torch.float)
                    self.priv_buf[i,3] = torch.tensor(obj_restitution[0],device=self.rl_device,dtype=torch.float)
            
            # if dr_config['randomize_object_scale']:
            #     w = randomize_object_scale(self.gym, self.envs[i], self.object_handles[i],dr_config['object_scale_config'])
            #     self.object_scales[i] = w
                
            if dr_config['randomize_object_mass']:
                mass = randomize_object_mass(self.gym, self.envs[i], self.object_handles[i],dr_config['object_mass_config'])
                
                if not self.cfg["env"]["addZerosInPrivBuf"]:
                    self.priv_buf[i,4] =torch.tensor(mass[0],device=self.rl_device,dtype=torch.float)
                
            # if dr_config['randomize_table_position']:
            #     randomize_table_z(self.gym, self.envs[i], self.table_handles[i],dr_config['table_scale_config'])
        
        return 
