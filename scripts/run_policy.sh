POLICY=$1 #"outputs/AllegroXarmGrasping_scratch_vel_control/2024-05-29_00-49/stage1_nn/ep_41700_step_1708M_reward_1876.28.pth"
cmd="python scripts/finetune.py   num_gpus=1 \
    task=AllegroXarmNew test=True headless=False \
    task.env.useOldActionSpace=False \
    checkpoint=$POLICY  \
    train.algo=PPOTransformer \
    wandb_activate=False  wandb_name=AllegroXarmGrasping_Finetuned \
    pipeline=gpu  rl_device=cuda:0  sim_device=cuda:0 \
    train.ppo.minibatch_size=16 num_envs=16 \
    task.env.episodeLength=600 \
    task.env.maxConsecutiveSuccesses=1 \
    pc_input=True \
    seed=-1"

echo $cmd
eval $cmd
