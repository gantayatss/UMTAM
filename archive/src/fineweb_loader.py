"""
Improved FineWeb Dataset Loader
================================

Handles both saved datasets and streaming from HuggingFace.
"""

from datasets import load_dataset, load_from_disk
from pathlib import Path
import torch


class FineWebDataset:
    """
    Flexible FineWeb dataset loader.
    
    Supports:
    1. Loading from saved disk location (recommended)
    2. Streaming from HuggingFace cache
    3. Downloading and caching
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        streaming: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.streaming = streaming
        
        data_path = Path(data_path)
        
        # Check if it's a saved dataset directory (has .arrow files)
        if data_path.exists() and (data_path / 'dataset_info.json').exists():
            print(f"✅ Loading saved dataset from: {data_path}")
            self.dataset = load_from_disk(str(data_path))
            self.streaming = False  # Can't stream from saved dataset
            print(f"   Loaded {len(self.dataset)} samples")
            
        elif data_path.exists() and list(data_path.glob('*.arrow')):
            # Has .arrow files but no dataset_info.json
            # This is the cache directory
            print(f"✅ Loading from cache: {data_path}")
            self.dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                cache_dir=str(data_path),
                streaming=streaming,
            )
            
        else:
            # Download and cache
            print(f"📥 Downloading FineWeb to: {data_path}")
            print(f"   (This may take a while...)")
            self.dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                cache_dir=str(data_path),
                streaming=streaming,
            )
    
    def __iter__(self):
        """Iterate through dataset and yield tokenized samples."""
        for sample in self.dataset:
            text = sample['text']
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
    
    def __len__(self):
        """Return dataset length if available."""
        if self.streaming:
            return None  # Unknown for streaming
        try:
            return len(self.dataset)
        except:
            return None


# For backwards compatibility
def create_fineweb_dataset(data_dir, tokenizer, max_length=1024, streaming=True):
    """
    Convenience function to create FineWeb dataset.
    
    Args:
        data_dir: Path to dataset (saved or cache directory)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        streaming: Whether to stream (only works with cache, not saved datasets)
    
    Returns:
        FineWebDataset instance
    """
    return FineWebDataset(data_dir, tokenizer, max_length, streaming)


if __name__ == '__main__':
    # Test the loader
    from transformers import GPT2Tokenizer
    
    print("Testing FineWeb Dataset Loader")
    print("="*60)
    
    # Test with your actual path
    data_path = "/Volumes/LaCie/Alirezas/umtam/fineweb_10bt_saved"
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = FineWebDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512,
        streaming=False,  # Use saved data
    )
    
    # Test iteration
    print("\nTesting data loading:")
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"\nSample {i}:")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  First 50 tokens: {sample['input_ids'][:50]}")
    
    print("\n✅ Dataset loader works!")
