#!/bin/bash 
#SBATCH --partition=mmaire-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=03:55:00
#SBATCH --output=/home/zsarwar/slurm/out/p%j.%N.stdout
#SBATCH --error=/home/zsarwar/slurm/out/%j.%N.stderr
#SBATCH --job-name=lore
#SBATCH --gres=gpu:a40:4

conda activate /net/projects/mmairegroup/zsarwar/c_envs/c_neox
cd /home/zsarwar/Projects/gpt_neox/gpt-neox/c2_configs/bash_scripts/
./lora.sh

