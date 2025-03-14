#!/bin/bash
#SBATCH --partition=peanut-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=/home/zsarwar/slurm/out/p%j.%N.stdout
#SBATCH --error=/home/zsarwar/slurm/out/%j.%N.stderr
#SBATCH --job-name=hf


conda activate /net/projects/mmairegroup/zsarwar/c_envs/c_gptneox

# ./tokenize.sh


# cd /u/zsarwar/gpt-neox/gpt-neox/code/
# python3 prep_fineweb_edu.py
python3 prep_json_data.py