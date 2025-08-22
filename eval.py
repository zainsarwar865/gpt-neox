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
import datasets


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
    # manually set the master port
    import time
    import random
    # os.environ["MASTER_PORT"] = str(10000 + int(random.randint(0, 10000)))
    model, neox_args = setup_for_inference_or_eval(
        use_cache=False, input_args=input_args, overwrite_values=overwrite_values
    )
    # print_rank_0(neox_args.eval_tasks)
    eval_tasks = ['truthfulqa_mc1', 'pubmedqa', 'arc_easy', 'sciq','winogrande', 'cola', 'hellaswag', 'mmlu','lambada_openai', 'arc_challenge', 'openbookqa', 'boolq', 'mnli']

    print_rank_0(eval_tasks)
    for x in eval_tasks:
        print_rank_0('Running task:', x)
        results = run_eval_harness(
            model,
            forward_step,
            neox_args,
            eval_tasks=[x],
            bootstrap_iters=10000,
            num_fewshot=5,
        )

        print_rank_0("After task:", x)
        try:
            print_rank_0(results["results"][x])
        except:
            print_rank_0(results)
        
        # Save results immediately for each task
        if neox_args.rank == 0:
            eval_name = list(results["results"].keys())[0]
            print('results')
            pprint(results['results'])
            exp_tag = neox_args.load.split('/')[-1]
            eval_results_dir = f'GPT_experts-8-topk-1-layers8-heads-32-lora' 
            os.makedirs(eval_results_dir, exist_ok=True)
            results_path = (
                f'{eval_results_dir}/{eval_name}_{exp_tag}.json'
            )
            print(f"Saving results to {results_path}")
            if neox_args.eval_results_prefix:
                results_path = f"{neox_args.eval_results_prefix}_{results_path}"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
                
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
