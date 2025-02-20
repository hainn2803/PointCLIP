#!/bin/bash -lex
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=slice
#SBATCH --output=/lustre/scratch/client/movian/research/users/hainn14/PointCLIP/sbatch/slurm_%A.log
#SBATCH --error=/lustre/scratch/client/movian/research/users/hainn14/PointCLIP/sbatch/slurm_%A.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=50GB
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.hainn14@vinai.io

# (optional) debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=bond0

   
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"

conda deactivate
conda activate /lustre/scratch/client/movian/research/users/hainn14/envs/pc

#!/bin/bash
cd ..

# Path to dataset
DATA=data/modelnet40_ply_hdf5_2048
DATASET=modelnet40

# PointCLIP_ZS or PointCLIP_FS
TRAINER=PointCLIP_FS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=rn101

CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
NUM_PROMPTS=4  # number of proxy
LOGIT_SCALE=on # on off

# Shot number
NUM_SHOTS=16

export CUDA_VISIBLE_DEVICES=0,1,2,4,5
python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--num-shots ${NUM_SHOTS} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output_spp/PROMPT_LR_0.08/ADAPTER_LR_0.01/NUM_SHOTS_${NUM_SHOTS}/NUM_PROMPTS_${NUM_PROMPTS}/PLOT_learnable_prompt_logit_scale_${LOGIT_SCALE}/${TRAINER}/${CFG}/${DATASET} \
--num-prompts ${NUM_PROMPTS} \
--logit-scale ${LOGIT_SCALE} \
--post-search