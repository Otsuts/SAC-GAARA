# SAC-GAARA
This is a RL project of course principle of AI on dm_control.Two students in our group are Jizheng Chen and Junru Gong.Part of code comes from https://github.com/denisyarats/pytorch_sac_ae

GAARA:Generate Amplified And Reconstructable latent state with Ae

The name GAARA pays tribute to the beloved character in cartoon Naruto Shipudden
![./](gaara.jpg)


To train an SAC+AE agent on the cheetah run task follow our final test parameter setting run:
```
python train.py     --domain_name walker     --task_name walk     --encoder_type pixel     --decoder_type pixel     --action_repeat 4     --save_video     --save_tb  --save_model  --work_dir ./log/walker_0.5 --seed 1   
```


```
usage: train.py [-h] [--pre_transform_image_size PRE_TRANSFORM_IMAGE_SIZE]
                [--domain_name DOMAIN_NAME] [--task_name TASK_NAME]
                [--image_size IMAGE_SIZE] [--action_repeat ACTION_REPEAT]
                [--frame_stack FRAME_STACK]
                [--replay_buffer_capacity REPLAY_BUFFER_CAPACITY] [--agent AGENT]
                [--init_steps INIT_STEPS] [--num_train_steps NUM_TRAIN_STEPS]
                [--batch_size BATCH_SIZE] [--hidden_dim HIDDEN_DIM]
                [--eval_freq EVAL_FREQ] [--num_eval_episodes NUM_EVAL_EPISODES]
                [--critic_lr CRITIC_LR] [--critic_beta CRITIC_BETA]
                [--critic_tau CRITIC_TAU]
                [--critic_target_update_freq CRITIC_TARGET_UPDATE_FREQ]
                [--actor_lr ACTOR_LR] [--actor_beta ACTOR_BETA]
                [--actor_log_std_min ACTOR_LOG_STD_MIN]
                [--actor_log_std_max ACTOR_LOG_STD_MAX]
                [--actor_update_freq ACTOR_UPDATE_FREQ]
                [--encoder_type ENCODER_TYPE]
                [--encoder_feature_dim ENCODER_FEATURE_DIM]
                [--encoder_lr ENCODER_LR] [--encoder_tau ENCODER_TAU]
                [--decoder_type DECODER_TYPE] [--decoder_lr DECODER_LR]
                [--decoder_update_freq DECODER_UPDATE_FREQ]
                [--decoder_latent_lambda DECODER_LATENT_LAMBDA]
                [--decoder_weight_lambda DECODER_WEIGHT_LAMBDA]
                [--comparison_lambda COMPARISON_LAMBDA] [--num_layers NUM_LAYERS]
                [--num_filters NUM_FILTERS] [--curl_latent_dim CURL_LATENT_DIM]
                [--discount DISCOUNT] [--init_temperature INIT_TEMPERATURE]
                [--alpha_lr ALPHA_LR] [--alpha_beta ALPHA_BETA] [--seed SEED]
                [--work_dir WORK_DIR] [--save_tb] [--save_model] [--save_buffer]
                [--save_video]

optional arguments:
  -h, --help            show this help message and exit
  --pre_transform_image_size PRE_TRANSFORM_IMAGE_SIZE
  --domain_name DOMAIN_NAME
  --task_name TASK_NAME
  --image_size IMAGE_SIZE
  --action_repeat ACTION_REPEAT
  --frame_stack FRAME_STACK
  --replay_buffer_capacity REPLAY_BUFFER_CAPACITY
  --agent AGENT
  --init_steps INIT_STEPS
  --num_train_steps NUM_TRAIN_STEPS
  --batch_size BATCH_SIZE
  --hidden_dim HIDDEN_DIM
  --eval_freq EVAL_FREQ
  --num_eval_episodes NUM_EVAL_EPISODES
  --critic_lr CRITIC_LR
  --critic_beta CRITIC_BETA
  --critic_tau CRITIC_TAU
  --critic_target_update_freq CRITIC_TARGET_UPDATE_FREQ
  --actor_lr ACTOR_LR
  --actor_beta ACTOR_BETA
  --actor_log_std_min ACTOR_LOG_STD_MIN
  --actor_log_std_max ACTOR_LOG_STD_MAX
  --actor_update_freq ACTOR_UPDATE_FREQ
  --encoder_type ENCODER_TYPE
  --encoder_feature_dim ENCODER_FEATURE_DIM
  --encoder_lr ENCODER_LR
  --encoder_tau ENCODER_TAU
  --decoder_type DECODER_TYPE
  --decoder_lr DECODER_LR
  --decoder_update_freq DECODER_UPDATE_FREQ
  --decoder_latent_lambda DECODER_LATENT_LAMBDA
  --decoder_weight_lambda DECODER_WEIGHT_LAMBDA
  --comparison_lambda COMPARISON_LAMBDA
  --num_layers NUM_LAYERS
  --num_filters NUM_FILTERS
  --curl_latent_dim CURL_LATENT_DIM
  --discount DISCOUNT
  --init_temperature INIT_TEMPERATURE
  --alpha_lr ALPHA_LR
  --alpha_beta ALPHA_BETA
  --seed SEED
  --work_dir WORK_DIR
  --save_tb
  --save_model
  --save_buffer
  --save_video
  ```

