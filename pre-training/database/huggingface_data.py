import os
from datetime import datetime
from datasets import load_dataset
from config import LOCAL_DATASET_FOLDER

DATASET_NAME = "range3/cc100-ja"

print(f"Loading dataset '{DATASET_NAME}'...")
dataset = load_dataset(DATASET_NAME, split="train")
print("Dataset loaded successfully.")

# Create timestamped folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(LOCAL_DATASET_FOLDER, f"cc100ja_csv_{timestamp}")
os.makedirs(save_path, exist_ok=True)

# Number of CSV files to split into
NUM_SHARDS = 50

print(f"Saving dataset into {NUM_SHARDS} CSV shards...")

for i in range(NUM_SHARDS):
    print(f" → Writing shard {i+1}/{NUM_SHARDS} ...")
    shard = dataset.shard(num_shards=NUM_SHARDS, index=i)
    shard.to_csv(os.path.join(save_path, f"cc100_ja_{i}.csv"), index=False)

print(f"✅ Done! CSV files are saved in: {save_path}")
