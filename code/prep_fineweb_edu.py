# import os
# from datasets import load_dataset

# # Set Hugging Face Cache Paths
# os.environ["HF_HOME"] = "/net/scratch/zsarwar/Datasets/huggingface_datasets"
# os.environ["HF_DATASETS_CACHE"] = "/net/scratch/zsarwar/Datasets/huggingface_datasets"

# # Configurations
# TOKEN_LIMIT = 2_000_000_000  # 12B total tokens
# BATCH_SIZE = 1_000_00000  # Write every 1M tokens
# OUTPUT_FILE = "/net/scratch/zsarwar/Datasets/Fineweb/raw/fineweb_edu_raw_2B.txt"
# DATASET_NAME = "HuggingFaceFW/fineweb-edu"
# DATASET_SPLIT = "train"

# # Ensure output directory exists
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# # Load dataset in streaming mode
# dataset = load_dataset(DATASET_NAME, name="CC-MAIN-2024-10", split=DATASET_SPLIT, streaming=True)

# # Open file for writing
# written_tokens = 0  # Total written tokens
# batch_tokens = 0  # Tracks tokens per batch
# buffer = []

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for example in dataset:
#         text = example["text"]
#         approx_tokens = len(text.split())  # Estimate token count

#         buffer.append(text)
#         batch_tokens += approx_tokens
#         written_tokens += approx_tokens  # Total progress

#         # Write to file when batch size is reached
#         if batch_tokens >= BATCH_SIZE:
#             f.write("\n".join(buffer) + "\n")
#             buffer = []  # Clear buffer to free RAM
#             batch_tokens = 0  # Reset batch counter
#             print(f"✅ {written_tokens} tokens written so far.")

#         # Stop when we reach 12B tokens
#         if written_tokens >= TOKEN_LIMIT:
#             break

#     # Write any remaining text in the buffer
#     if buffer:
#         f.write("\n".join(buffer) + "\n")

# print(f"✅ Finished! Saved {written_tokens} tokens to {OUTPUT_FILE}")







import os
from datasets import load_dataset
import itertools

# Set Hugging Face Cache Paths
os.environ["HF_HOME"] = "/net/scratch/zsarwar/Datasets/huggingface_datasets"
os.environ["HF_DATASETS_CACHE"] = "/net/scratch/zsarwar/Datasets/huggingface_datasets"


# Configurations
TOKEN_LIMIT = 500_000_000  # 1B total tokens
SKIP_TOKENS = 2_000_000_000  # Skip first 12B tokens
BATCH_SIZE = 10_000_0000  # Write every 10M tokens
OUTPUT_FILE = "/net/scratch/zsarwar/Datasets/Fineweb/raw/fineweb_edu_raw_val_500M.txt"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT = "train"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load dataset in streaming mode
dataset = load_dataset(DATASET_NAME, name="CC-MAIN-2024-10", split=DATASET_SPLIT, streaming=True)

# Skip the first 12B tokens manually (using itertools to discard)
dataset_iter = iter(dataset)
discard_tokens = 0
print(f"⏳ Skipping first {SKIP_TOKENS} tokens...")
for idx, example in enumerate(dataset_iter):
    if idx % 100000 == 0:
        print(f"Skipped {idx}")
    discard_tokens += len(example["text"].split())  # Approximate token count
    if discard_tokens >= SKIP_TOKENS:
        print(f"✅ Skipped {discard_tokens} tokens. Starting collection.")
        break

# Start writing the next 1B tokens
written_tokens = 0
batch_tokens = 0
buffer = []

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for example in dataset_iter:
        text = example["text"]
        approx_tokens = len(text.split())  # Estimate token count

        buffer.append(text)
        batch_tokens += approx_tokens
        written_tokens += approx_tokens  # Total progress

        # Write to file when batch size is reached
        if batch_tokens >= BATCH_SIZE:
            f.write("\n".join(buffer) + "\n")
            buffer = []  # Clear buffer to free RAM
            batch_tokens = 0  # Reset batch counter
            print(f"✅ {written_tokens} tokens written so far.")

        # Stop when we reach 1B tokens
        if written_tokens >= TOKEN_LIMIT:
            break

    # Write any remaining text in the buffer
    if buffer:
        f.write("\n".join(buffer) + "\n")

print(f"✅ Finished! Saved {written_tokens} tokens to {OUTPUT_FILE}")