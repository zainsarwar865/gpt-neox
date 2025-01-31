
export TRAIN_PATH='/fsx-claim/vox781/repos/neox/gpt-neox-base-moe/'
cd $TRAIN_PATH

export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export HF_HOME="/fsx-claim/sambits_interns_2024/data/eval_harness_tasks_data/"
export FSX_EVAL_DATASET_PATH="/fsx-claim/sambits_interns_2024/data/eval_harness_tasks_data/"
export DATA_PATH="/fsx-claim/sambits_interns_2024/data/eval_harness_tasks_data/"
export data_path="/fsx-claim/sambits_interns_2024/data/eval_harness_tasks_data/"
export NLTK_DATA="/fsx-claim/uls607/gpt_neox_install/gpt-neox/tokenizers"
export data_dir=""

python ./deepy.py eval.py -d c1_configs \
eval_harness_fw.yml \
--eval_tasks mmlu_high_school_geography
#--eval_tasks hellaswag ARC-Challenge ARC-Easy boolq copa openbookqa piqa sciq winogrande_xl winogrande lambada_standard

# --eval_tasks "hellaswag arc_challenge arc_easy boolq copa evaluate openbookqa pinpoint_qa piqa sciq truthfulqa truthful_qa_mc winogrande_xl"