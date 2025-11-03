"""
data.py

Load the samsum dataset from HuggingFace, tokenize with a pretrained tokenizer (e.g., "t5-small" tokenizer),
and prepare PyTorch DataLoader objects for training/validation.

Note: This file assumes internet access when running the training script to download the dataset and tokenizer.
"""

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

class SummarizationDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="t5-small", max_source_length=256, max_target_length=64, sample_size=None):
        ds = load_dataset("knkarthick/samsum", split=split)
        if sample_size is not None:
            ds = ds.select(range(min(sample_size, len(ds))))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item["dialogue"]
        tgt = item["summary"]
        tokenized_src = self.tokenizer(src, truncation=True, padding="max_length", max_length=self.max_source_length, return_tensors="pt")
        tokenized_tgt = self.tokenizer(tgt, truncation=True, padding="max_length", max_length=self.max_target_length, return_tensors="pt")
        return {
            "input_ids": tokenized_src["input_ids"].squeeze(0),
            "attention_mask": tokenized_src["attention_mask"].squeeze(0),
            "labels": tokenized_tgt["input_ids"].squeeze(0)
        }

def collate_fn(batch):
    # batch is a list of dicts
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def get_dataloaders(tokenizer_name="t5-small", batch_size=16, max_source_length=256, max_target_length=64, sample_size=200):
    train_ds = SummarizationDataset(split="train", tokenizer_name=tokenizer_name, max_source_length=max_source_length, max_target_length=max_target_length, sample_size=sample_size)
    val_ds = SummarizationDataset(split="validation", tokenizer_name=tokenizer_name, max_source_length=max_source_length, max_target_length=max_target_length, sample_size= min(50, sample_size//5))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, train_ds.tokenizer
