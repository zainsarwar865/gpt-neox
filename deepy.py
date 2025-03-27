#!/usr/bin/env python
# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

import logging
import os
os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
os.environ['RANK'] = str(0)
os.environ['NCCL_P2P_DISABLE']=str(1)
import deepspeed.launcher.runner


def main(input_args=None):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    from megatron.neox_arguments import NeoXArgs
    from megatron.utils import get_wandb_api_key

    neox_args = NeoXArgs.consume_deepy_args(input_args)

    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    # Extract wandb API key and inject into worker environments
    wandb_token = get_wandb_api_key(neox_args=neox_args)
    if wandb_token is not None:
        deepspeed.launcher.runner.EXPORT_ENVS.append("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_token

    deepspeed.launcher.runner.main(deepspeed_main_args)


if __name__ == "__main__":
    main()
