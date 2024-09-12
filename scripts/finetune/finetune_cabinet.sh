cmd="python scripts/finetune.py   num_gpus=8 \
    checkpoint="outputs/Policy_noise01_l4h4_ctx_16_data_mix_simrob/dt_25-05-2024_07-02-31/model_step_831207.pt"\
    task=AllegroXarmCabinet \
    train.algo=PPOTransformer \
    train.ppo.initEpsHand=0.5 \
    train.ppo.initEpsArm=0.5 \
    train.ppo.value_grads_to_pointnet=False \
    train.ppo.critic_warmup_steps=200 \
    train.ppo.learning_rate=1e-5 \
    wandb_activate=True  wandb_name=AllegroXarmCabinet_finetune_datamix_pretraining_initeps_0.5 \
    pipeline=gpu  rl_device=cuda:0  sim_device=cuda:0 \
    train.ppo.minibatch_size=512 num_envs=512 \
    seed=-1"

echo $cmd
eval $cmd
