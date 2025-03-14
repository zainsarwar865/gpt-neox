#!/bin/bash
#SBATCH --partition=peanut-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=/home/zsarwar/slurm/out/p%j.%N.stdout
#SBATCH --error=/home/zsarwar/slurm/out/%j.%N.stderr
#SBATCH --job-name=hf


conda activate /net/projects/mmairegroup/zsarwar/c_envs/c_neox

cd /home/zsarwar/Projects/neox/gpt-neox-base/tools/datasets
python3  preprocess_data.py \
--input "/net/scratch/zsarwar/Datasets/Fineweb/raw/fineweb_edu_raw_val_500M.jsonl" \
--tokenizer-type HFTokenizer \
--vocab-file "/net/projects/mmairegroup/zsarwar/tokenizers/llama3_tokenizer.json" \
--output-prefix "/net/scratch/zsarwar/Datasets/Fineweb/tokenized/tokenized_val/tokenized_fineweb_edu_500M" \
--workers 8 \
--log-interval 1000
