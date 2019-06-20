#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --do_train --cuda \
     --do_valid \
    --do_test \
    --data_path data/FB15k \
    --model RotatE \
    -n 256 -b 512 -d 1000 \
    -g 24.0 -a 1.0 -adv \
    -lr 0.0001 --max_steps 150000 \
    -save models/RotatE_FB15k_0 --test_batch_size 16 -de
