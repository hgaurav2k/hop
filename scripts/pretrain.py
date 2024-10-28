import isaacgym 
from tasks import isaacgym_task_map
import torch 
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
import wandb
from torch.optim import Adam, AdamW
from algo.pretrained.trainer import RobotTrainer 
import wandb 
from algo.pretrained.robot_transformer_ar import RobotTransformerAR
from algo.pretrained.robot_dataset import RobotDataset , collate_fn
import os 
from datetime import datetime 
import json 
import hydra 
from utils.reformat import omegaconf_to_dict, print_dict
from utils.utils import set_np_formatting, set_seed
from utils.logger import Logger
import random 
import numpy as np 
from torch.optim.lr_scheduler import CosineAnnealingLR 

@hydra.main(config_name='config', config_path='../cfg/')
def main(config: DictConfig):


    device = config.pretrain.device  
    config.seed = set_seed(config.seed)

    capture_video = config.task.env.enableVideoLog

    if config.pretrain.wandb_activate:
        wandb.init(project="manipulation-pretraining",
                    name=config.pretrain.wandb_name,
                  config=omegaconf_to_dict(config))
        
    tmodel = RobotTransformerAR
    
    if config.pretrain.test: 

        model = tmodel(
            cfg=config
        )

        model = model.to(device)

        model.eval()

        assert config.pretrain.checkpoint != ''
           # set numpy formatting for printing only
        set_np_formatting()


        if config.pretrain.wandb_activate:
            wandb_logger = wandb.init(project=config.wandb_project, 
                                    name=config.pretrain.wandb_name,
                                    entity=config.wandb_entity, 
                                    config=omegaconf_to_dict(config),
                                    sync_tensorboard=True)
        else:
            wandb_logger=None

        output_dif = os.path.join('outputs', config.wandb_name)
        logger = Logger(output_dif, summary_writer=wandb_logger)

        cprint('Start Building the Environment', 'green', attrs=['bold'])
    
        env = isaacgym_task_map[config.task_name](
            cfg=omegaconf_to_dict(config.task),
            pretrain_cfg=omegaconf_to_dict(config.pretrain),
            rl_device = config.rl_device,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
            virtual_screen_capture=config.capture_video,
            force_render=config.force_render
        )

        model.load_state_dict(torch.load(config.pretrain.checkpoint,map_location=device))
        
        cprint(f"Model loaded from {config.pretrain.checkpoint}", color='green', attrs=['bold'])

        model.run_multi_env(env, cfg=config)

        return 

    else:

        if config.pretrain.wandb_activate:
            wandb_logger = wandb.init(project=config.wandb_project, name=config.wandb_name,
                                       entity=config.wandb_entity, config=omegaconf_to_dict(config))
        else:
            wandb_logger=None

        train_dataset = RobotDataset(cfg=config, root=config.pretrain.training.root_dir)
        val_dataset = RobotDataset(cfg=config, root=config.pretrain.validation.root_dir)
        
        max_ep_len = max(train_dataset.max_ep_len, val_dataset.max_ep_len)

        cprint(f"Dataloader built", color='green', attrs=['bold'])

        model = tmodel(
            cfg=config,
            max_ep_len=max_ep_len
        )

        model = model.to(device)

        if config.pretrain.training.model_save_dir is not None:
                save_dir = config.pretrain.training.model_save_dir
                # Create the saving directory using the wandb name and the date and time
                os.makedirs(save_dir, exist_ok=True)
                #get date and time 
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                experiment_folder = os.path.join(save_dir, f'{config.pretrain.wandb_name}', f'dt_{dt_string}')
                # create the experiment folder if not exists
                os.makedirs(experiment_folder, exist_ok=True)
                json.dump(OmegaConf.to_container(config), open(os.path.join(experiment_folder, 'config.json'), 'w'))
                logger = Logger(experiment_folder, summary_writer=wandb_logger)

        else:
            save_dir = None
            logger = None 

        cprint(f"Model built", color='green', attrs=['bold'])

        if config.pretrain.training.load_checkpoint:
            assert os.path.exists(config.pretrain.checkpoint), f"Checkpoint {config.pretrain.checkpoint} does not exist"
            model.load_state_dict(torch.load(config.pretrain.checkpoint,map_location=device))
            model.train()
        cprint(f"Model loaded from {config.pretrain.checkpoint}", color='green', attrs=['bold'])

        scheduler = None #CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
        optimizer = AdamW(model.parameters(), lr=config.pretrain.training.lr, weight_decay=config.pretrain.training.weight_decay)
        loss_fn = torch.nn.L1Loss() #torch.nn.MSELoss()

        trainer = RobotTrainer(
            model = model,
            optimizer = optimizer, 
            scheduler = scheduler,
            train_dataset = train_dataset,
            val_dataset = val_dataset,
            collate_fn=collate_fn, 
            loss_fn = loss_fn,
            model_save_dir = experiment_folder,
            logger = logger,
            config=config
        )

        if capture_video:
            assert config.pretrain.wandb_activate, "Video capture requires wandb activation"
            # create the environment to capture the video
            env = isaacgym_task_map[config.task_name](
                cfg=omegaconf_to_dict(config.task),
                pretrain_cfg=omegaconf_to_dict(config.pretrain),
                rl_device = config.pretrain.device,
                sim_device=config.pretrain.device,
                graphics_device_id=config.graphics_device_id,
                headless=config.headless,
                virtual_screen_capture=config.capture_video,
                force_render=config.force_render
            )
        
        for i in range(config.pretrain.training.num_epochs):
            cprint("Training iteration {}".format(i), color='magenta', attrs=['bold'])
            outputs = trainer.train_epoch(iter_num=i, 
                                          print_logs=True)
            if config.pretrain.wandb_activate:
                wandb.log(outputs, commit=True)

            
            if capture_video:
                fps = int(1/(config.task.sim.dt*config.task.env.controlFrequencyInv))
                print(f"Capturing video from simulation")
                env.start_video_recording()
                info_dict = model.run_multi_env(env, cfg=config)
                video_frames = env.stop_video_recording()
                logger.log_video(video_frames, name="Test Performance", fps=fps, step=(i+1)*len(trainer.train_dataloader))
                logger.log_dict(info_dict, (i+1)*len(trainer.train_dataloader), verbose=False)
                env.video_frames = []



if __name__ == '__main__':
    main()
