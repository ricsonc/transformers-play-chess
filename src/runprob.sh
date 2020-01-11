#!/usr/bin/env bash

# t2t-datagen \
#   --t2t_usr_dir=. \
#   --data_dir=data \
#   --tmp_dir=tmp \
#   --problem=chess

# t2t-trainer \
#     --t2t_usr_dir=. \
#     --data_dir=data \
#     --model=transformer \
#     --hparams_set=transformer_big_single_gpu \
#     --problem=chess \
#     --tmp_dir=tmp \
#     --output_dir=out \
#     --train_steps=1000000000 \
#     --worker_gpu_memory_fraction=0.9

t2t-decoder \
    --t2t_usr_dir=. \
    --data_dir=data \
    --problem=chess \
    --model=transformer \
    --output_dir=out \
    --hparams_set=transformer_big_single_gpu \
    --worker_gpu_memory_fraction=0.9 \
    --hparams='sampling_method=random,sampling_temp=1.0' \
    --decode_hparams="beam_size=1" \
    --decode_interactive
