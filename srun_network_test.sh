#!/usr/bin/env bash
eps=300
batchsz=20
baselr=0.1
server=BJ-IDC1-10-10-15-74
gpus=4
augment=4

srun -p Med -n1 -w $server --mpi=pmi2 --gres=gpu:$gpus --ntasks-per-node=$gpus --job-name=brats2018 \
--kill-on-bad-exit=1 \
python network_test.py

