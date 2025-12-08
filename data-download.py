from datasets import load_dataset

# Download 100BT sample to your disk
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    cache_dir="./fineweb_data"  # Store in current directory
)

# Verify it's on disk
print(f"Dataset cached at: {dataset.cache_files}")
print(f"Number of samples: {len(dataset)}")

# Save as arrow format (efficient for ML training)
dataset.save_to_disk("/Users/alirezamoayedikia/Documents/Research/AI/fineweb_100bt_saved") # ./fineweb_100bt_saved")