#!/bin/bash
cd ..

# Path to dataset
DATA=data/modelnet40_ply_hdf5_2048
DATASET=modelnet40
NUM_CLASSES=40

# PointCLIP_ZS or PointCLIP_FS
TRAINER=PointCLIP_FS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=rn101

CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
N=1  # number of proxy

# Shot number
NUM_SHOTS=16

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--num-shots ${NUM_SHOTS} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output_N_4_learnable_prompt_adapter/${TRAINER}/${CFG}/${DATASET} \
--num-classes ${NUM_CLASSES} \
--post-search