conda activate /u/zsarwar/c_envs/c_neox
cd /u/zsarwar/gpt-neox/gpt-neox/tools/datasets

python3  preprocess_data.py \
--input "/scratch/bdgs/zsarwar/datasets/raw/fineweb_edu_2B_val.jsonl" \
--tokenizer-type HFTokenizer \
--vocab-file "/u/zsarwar/data/tokenizers/llama3_tokenizer.json" \
--output-prefix "/scratch/bdgs/zsarwar/datasets/tokenized" \
--workers 64 \
--log-interval 10000