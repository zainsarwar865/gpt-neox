# global_log.py
import collections

tokens_per_lora_log = collections.defaultdict(list)  # key = layer_num, value = list of tensors
forward_step_counter = 0