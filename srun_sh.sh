#!/usr/bin/env bash
eps=300
batchsz=20
baselr=0.1
server=SH-IDC1-10-5-34-117
gpus=8
augment=4

srun -p MIA -n1 -w $server --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=brats2018 \
--kill-on-bad-exit=1 \
python train_segmentation_BRATS.py  -c ./configs/config_unet_ct_dsv_brats_server.json