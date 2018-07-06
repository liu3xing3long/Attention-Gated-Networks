#!/usr/bin/env bash
server=BJ-IDC1-10-10-15-73
gpus=8

srun -p Med -n1 -w $server --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=brats2018 \
--kill-on-bad-exit=1 \
python train_segmentation_BRATS.py  -c ./configs/config_unet_ct_dsv_brats_server.json

