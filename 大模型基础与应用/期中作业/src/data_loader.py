from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class SAMSumDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 编码对话（输入）
        input_encoding = self.tokenizer(
            item['dialogue'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 编码摘要（目标）
        target_encoding = self.tokenizer(
            item['summary'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }


def get_data_loaders(config):
    """获取数据加载器"""
    # 加载数据集
    dataset = load_dataset(config.dataset_name)

    # 加载tokenizer（使用BART的tokenizer）
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    # 创建数据集实例
    train_dataset = SAMSumDataset(
        dataset['train'], tokenizer, config.max_input_length, config.max_target_length
    )
    val_dataset = SAMSumDataset(
        dataset['validation'], tokenizer, config.max_input_length, config.max_target_length
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    return train_loader, val_loader, tokenizer