cd /u/zsarwar/gpt-neox/gpt-neox/tools/datasets/
python3  preprocess_data.py \
--input "/scratch/bdgs/zsarwar/datasets/raw/fineweb_edu_12B.jsonl" \
--tokenizer-type HFTokenizer \
--vocab-file "/u/zsarwar/data/tokenizers/llama3_tokenizer.json" \
--output-prefix "/scratch/bdgs/zsarwar/datasets/tokenized_train/tokenized_train_12b" \
--workers 64 \
--log-interval 10000
