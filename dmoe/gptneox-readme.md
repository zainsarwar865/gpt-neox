# Readme to run gptneox


# Setup a conda environment

conda create --prefix `pwd`/c_gptneox python==3.8.19


## Clone GPTNeoX
git clone https://github.cloud.capitalone.com/VOX781/gpt-neox.git

cd gpt-neox

# switch to base branch

git checkout base-moe

# Install requirements

pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-moe.txt


# Change Deepspeed code to add torchrun

git clone https://github.cloud.capitalone.com/VOX781/DeepSpeedX.git

# cd into deepspeed 

cd /fsx-claim/vox781/c_envs/c_gptneox/lib/python3.8/site-packages/deepspeed

# Replace folder launcher with the launcher in the repo DeepSpeedX


# Do a test run

cd c1_configs/bash_scripts

./moe_125M_8c1_3072_3e-4.sh

# The first run will be slow as it will take a while to build the fused kernels


