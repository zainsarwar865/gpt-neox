# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Evaluation tasks - modified from https://github.com/EleutherAI/lm-evaluation-harness"""
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.training import forward_step
from megatron.utils import setup_for_inference_or_eval, init_wandb, print_rank_0
from megatron.logging import tb_wandb_log
from eval_tasks import run_eval_harness
from pprint import pprint
from datetime import datetime
import json
import torch


def main(input_args=None, overwrite_values=None):
    model, neox_args = setup_for_inference_or_eval(
        use_cache=False, input_args=input_args, overwrite_values=overwrite_values
    )

    print_rank_0(neox_args.eval_tasks)

    all_results = []
    for x in neox_args.eval_tasks:
        print_rank_0('Running task:', x)
        results = run_eval_harness(
            model,
            forward_step,
            neox_args,
            eval_tasks=[x],
            bootstrap_iters=10000,
            num_fewshot=5,
        )
        all_results.append(results)
        print_rank_0("After task:", x)
        torch.distributed.barrier()

    



    if neox_args.rank == 0:

        for results in all_results:
            init_wandb(neox_args=neox_args)

            eval_name = list(results["results"].keys())[0]


            # log to wandb
            for k, v in results["results"].items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        k3 = "_".join([k, k2])
                        tb_wandb_log(
                            f"eval/{k3}",
                            v2,
                            neox_args.iteration,
                            use_wandb=neox_args.use_wandb,
                        )
                else:
                    tb_wandb_log(
                        f"eval/{k}",
                        v,
                        neox_args.iteration,
                        use_wandb=neox_args.use_wandb,
                    )



            for key in ['results',]:# 'configs', 'versions', 'n-shot', 'git_hash']:
                print(key)
                pprint(results[key])
            # pprint(results)
            results_path = (
                f'eval_results_{eval_name}_{datetime.now().strftime("%m-%d-%Y-%H-%M-%S")}.json'
            )
            if neox_args.eval_results_prefix:
                results_path = f"{neox_args.eval_results_prefix}_{results_path}"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
