from datasets import load_dataset

# Stream from disk - no upfront loading!
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    cache_dir="/Volumes/LaCie/Alirezas/memory-optimization/fineweb_data",
    streaming=True  # ← Loads on-the-fly
)

print("✅ Ready instantly!")

# Access data as needed
for i, sample in enumerate(dataset):
    if i >= 3:
        break
    print(f"\n--- Sample {i} ---")
    print(f"Text: {sample['text']}")

