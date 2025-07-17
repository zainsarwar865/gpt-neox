import os
import sys

lm_eval_local_path = "/fsx/vox781/Code/" 
sys.path.insert(0, lm_eval_local_path) # Use insert(0, ...) to prioritize your local version

print(sys.path)
# from lm_eval.models.huggingface import HFLM
from lm_eval import tasks
# from lm_eval.models.utils import chunks