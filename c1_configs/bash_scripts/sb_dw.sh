#!/bin/bash
#SBATCH --partition=gpuA100x8
#SBATCH --account=bdgs-delta-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/u/zsarwar/logs/neox/p%j.%N.stdout
#SBATCH --error=/u/zsarwar/logs/neox/%j.%N.stderr
#SBATCH --job-name=torch
#SBATCH --gpus-per-node=8

conda activate /u/zsarwar/c_envs/c_neox
export WORLD_SIZE=4
module load cuda/12.4.0

debug_fw_edu.sh > a80.txt
