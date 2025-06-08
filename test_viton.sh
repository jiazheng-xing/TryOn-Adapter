#!/bin/bash

python test_viton.py --plms --gpu_id 3 \
--ddim_steps 100 \
--outdir new_results_GP/viton \
--config configs/viton.yaml \
--dataroot /users/ock/tryon-data/Vition-HD \
--ckpt /users/ock/TryOn-Adapter/models/ckpts/viton.ckpt \
--ckpt_elbm_path models/vae/dirs/viton_hd_blend \
--use_T_repaint True \
--n_samples 1 \
--seed 23 \
--scale 1 \
--H 512 \
--W 512 \
--unpaired
