#!/bin/bash

python test_dresscode.py --plms --gpu_id 3 \
--ddim_steps 100 \
--outdir new_results_GP/dresscode \
--config configs/dresscode.yaml \
--dataroot  /users/ock/tryon-data/DressCode \
--ckpt models/ckpts/dresscode.ckpt \
--ckpt_elbm_path models/vae/dirs/dresscode_blend \
--use_T_repaint True \
--n_samples 1 \
--seed 23 \
--scale 1 \
--H 512 \
--W 512 \
# --unpaired
