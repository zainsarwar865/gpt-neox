# The Complete Guide to setting up GPTNeoX

## Move to a base dir

## This guide assumes you are in some base directory we will call /base_dir

cd /base_dir

## Make required directories
mkdir -p gptneox-exps/gpt-neox repos/neox pkgs c_envs

## Environent variables
export MPICC=/opt/amazon/openmpi/bin/mpicc

export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

## Setup a conda environment in c_envs
cd c_envs

conda create --prefix `pwd`/c_gptneox python==3.8.19

## Install apex
cd ..

cd pkgs

git clone https://github.com/NVIDIA/apex

cd apex
### if pip >= 23.1 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
### otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## Clone GPTNeoX
cd ..

git clone https://github.cloud.capitalone.com/VOX781/gpt-neox.git

mv gpt-neox gpt-neox-base-moe

cd gpt-neox-base-moe

## switch to base-moe branch

git checkout base-moe

## Install requirements

pip install -r requirements/requirements.txt

pip install -r requirements/requirements-moe.txt


## We already installed deepspeed but now need to replace certain files in it to be able to use it with torchrun The changes are in the repository below

## Clone Modified DeepSpeed to work with Kubernetes
cd ..

cd pkgs

git clone https://github.cloud.capitalone.com/VOX781/DeepSpeedX.git


## Deepspeed is already installed now in the conda env
## cd into it 
## You may have to replace path with your own conda env path which we just made above
cd /base_dir/c_envs/c_gptneox/lib/python3.8/site-packages/deepspeed
## there is a folder called launcher. Delete it and replace with the version cloned in DeepSpeedX
rm -rf launcher

cp -r /base_dir/pkgs/DeepSpeedX/launcher .



## WE ARE DONE!

## cd back into gptneox
cd /base_dir/repos/neox/gpt-neox-base-moe
## Do a test run

cd c1_configs/bash_scripts

./debug.sh

## The first run will be slow as it will take a while to build the fused kernels




## That's fine, but where are the checkpoints being stored?
### They are being stored under gptneox-exps/gpt-neox with the following structure
### The meta folder will be made according to this config : "GPT_experts-{self.moe_num_experts}-topk-{self.moe_top_k}-layers{self.num_layers}-heads-{self.num_attention_heads}" which will be read from the yml file
### Each master and worker node will have their own directories made in the following format : {hostname}-{hash} where hash is a md5 hash of the entire yaml file. Therefore, it is critical that all jobs have a different yaml file else stuff will get overwritten. Any change in the yaml file or name of the yaml file will change the md5 hash and thus we will have a unique directory for every experiment.

## How to resume a job from a checkpoint in case it crashes or we need to run benchmarks?

## In the yaml file, there is a key : 'load' : 'none'. Replacing that with the checkpoint directory will continue the model from that checkpoint. 
## The checkpoint directory is stored in the master node for each job. There are three directories : logs, tensorboard and checkpoints. To load checkpoints, change the 'load' : 'none' key to 'load' : /checkpoints and it should work.



## GPTNEOX is stuck
## Sometimes, if the job crashes between the arguments being processed and the model being created, a lock will not be released and all future jobs will hang. Thus, it is a good idea to add this statement on the top of every bash file

rm /base_dir/repos/neox/gpt-neox-base-moe/megatron/fused_kernels/build/lock






## Which Branches to clone and what to name them?

## Zain : 
branch : base-moe : 

git clone https://github.cloud.capitalone.com/VOX781/gpt-neox.git

mv gpt-neox gpt-neox-base-moe

branch : lora : 

git clone https://github.cloud.capitalone.com/VOX781/gpt-neox.git

mv gpt-neox gpt-neox-base-lora

## Ashwinee : 

branch : dense-router : 

git clone https://github.cloud.capitalone.com/VOX781/gpt-neox.git

mv gpt-neox gpt-neox-dense-router
