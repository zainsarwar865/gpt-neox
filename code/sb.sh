#!/bin/bash
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bdgs-delta-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=06:30:00
#SBATCH --output=/u/zsarwar/logs/p%j.%N.stdout
#SBATCH --error=/u/zsarwar/logs/%j.%N.stderr
#SBATCH --job-name=torch
#SBATCH --gpus-per-node=1

conda activate /u/zsarwar/c_envs/c_neox

./tokenize.sh


# cd /u/zsarwar/gpt-neox/gpt-neox/code/
# python3 prep_fineweb_edu.py