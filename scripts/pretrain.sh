

DATADIR=$1
CMD="python scripts/pretrain.py num_gpus=1 headless=True \
    track_pose=False get_target_reference=False num_envs=25 \
    pc_input=True pipeline=cuda rl_device=cuda:0 sim_device=cuda:0 \
    pretrain.training.root_dir=$DATADIR/train \
    pretrain.validation.root_dir=$DATADIR/val pretrain.wandb_activate=True \
    pretrain.wandb_name=Policy_noise01_l4h4_ctx_16_data_mix_simrob seed=-1 \
    task.env.enableVideoLog=True \
    task.env.episodeLength=400"

echo $CMD 
eval $CMD