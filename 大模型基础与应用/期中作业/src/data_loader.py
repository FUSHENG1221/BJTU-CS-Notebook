import os
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from config import config

class SAMSumDataset(Dataset):
    """
    SAMSum对话摘要数据集的自定义Dataset类
    这个类负责将原始的SAMSum数据集转换为PyTorch可以处理的格式，
    包括文本分词、填充、截断等预处理操作。
    继承自torch.utils.data.Dataset，需要实现__len__和__getitem__方法。
    """

    def __init__(self, dataset, tokenizer, max_input_length=512, max_target_length=128):
        """
        初始化数据集
        Args:
            dataset: Hugging Face数据集对象，包含'id'、'dialogue'和'summary'字段
            tokenizer: 分词器，用于将文本转换为模型可处理的token ID序列
            max_input_length: 输入序列的最大长度（对话文本）
            max_target_length: 目标序列的最大长度（摘要文本）
        """
        self.dataset = dataset  # 原始数据集
        self.tokenizer = tokenizer  # 分词器实例
        self.max_input_length = max_input_length  # 输入文本最大长度
        self.max_target_length = max_target_length  # 目标文本最大长度

    def __len__(self):
        """
        返回数据集的样本数量
        这是Dataset类必须实现的方法，告诉DataLoader有多少个样本
        Returns:
            int: 数据集中样本的总数
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        这是Dataset类必须实现的核心方法，负责将原始数据转换为模型输入格式
        Args:
            idx (int): 样本索引
        Returns:
            dict: 包含以下键的字典：
                - input_ids: 分词后的对话文本ID序列
                - attention_mask: 注意力掩码，区分真实内容和填充部分
                - labels: 分词后的摘要文本ID序列（模型训练目标）
        """
        # 获取指定索引的样本
        if hasattr(self.dataset, 'iloc'):  # pandas DataFrame
            item = self.dataset.iloc[idx]
        else:  # Hugging Face Dataset
            item = self.dataset[idx]
        # item = self.dataset[idx]

        # 编码对话文本（模型输入）
        input_encoding = self.tokenizer(
            item['dialogue'],  # 原始对话文本
            max_length=self.max_input_length,  # 最大序列长度
            padding='max_length',  # 填充到最大长度（保证批次内长度一致）
            truncation=True,  # 启用截断（超长文本会被截断）
            return_tensors='pt'  # 返回PyTorch张量格式
        )
        # 参数说明：
        # - max_length: 控制序列长度，避免内存溢出
        # - padding='max_length': 将所有序列填充到相同长度，便于批次处理
        # - truncation=True: 自动截断超长文本
        # - return_tensors='pt': 返回PyTorch张量，而不是Python列表

        # 编码摘要文本（模型训练目标）
        target_encoding = self.tokenizer(
            item['summary'],  # 原始摘要文本
            max_length=self.max_target_length,  # 摘要通常比对话短
            padding='max_length',  # 同样需要填充到统一长度
            truncation=True,  # 启用截断
            return_tensors='pt'  # 返回PyTorch张量
        )

        # 返回预处理后的样本
        return {
            # 形状：(seq_len,)，因为tokenizer返回的是 (1, seq_len)，需要用 .squeeze() 移除自动添加的批次维度。
            'input_ids': input_encoding['input_ids'].squeeze(),  # 移除批次维度：(1, seq_len) -> (seq_len,)
            'attention_mask': input_encoding['attention_mask'].squeeze(),  # 注意力掩码
            'labels': target_encoding['input_ids'].squeeze()  # 训练标签（摘要的token ID）
        }
        # 字段说明：
        # - input_ids: 对话文本转换后的数字序列，形状为(seq_len,)
        # - attention_mask: 区分真实内容和填充部分，1=真实内容，0=填充部分
        # - labels: 摘要文本的数字序列，模型需要学习生成这个序列


def load_local_samsum_dataset(data_dir="../data/samsum"):
    """
    从本地CSV文件加载SAMSum数据集

    Args:
        data_dir: 本地数据目录路径，包含train.csv, validation.csv, test.csv

    Returns:
        DatasetDict: 包含train、validation、test分割的数据集字典
    """
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 定义CSV文件路径
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # 检查文件是否存在
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

    print("📁 从本地CSV文件加载SAMSum数据集...")

    # 读取CSV文件
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 检查必要的列是否存在
    required_columns = ['id', 'dialogue', 'summary']
    for df, split_name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"{split_name}缺少必要列: {col}")

    # 转换为Hugging Face Dataset格式
    # train_dataset = Dataset.from_pandas(train_df)
    # val_dataset = Dataset.from_pandas(val_df)
    # test_dataset = Dataset.from_pandas(test_df)

    # 创建DatasetDict
    # 直接返回pandas DataFrame，不使用Dataset.from_pandas
    dataset_dict = DatasetDict({
        'train': train_df,
        'validation': val_df,
        'test': test_df
    })
    # dataset_dict = DatasetDict({
    #     'train': train_dataset,
    #     'validation': val_dataset,
    #     'test': test_dataset
    # })

    return dataset_dict


def load_local_tokenizer(tokenizer_path="../data/tokenizer/facebook/bart-base"):
    """
    从本地加载分词器，支持多种回退方案
    Args:
        tokenizer_path: 本地分词器路径
    Returns:
        tokenizer: 分词器实例
    """
    if os.path.exists(tokenizer_path):
        try:
            print(f"🔍 尝试从本地加载分词器: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True  # 强制使用本地文件
            )
            print("✅ 成功从本地加载分词器")

            # 确保有必要的特殊token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token

            # print(f"分词器词汇表大小: {len(tokenizer)}")
            # print(f"填充token ID: {tokenizer.pad_token_id}")
            return tokenizer
        except Exception as e:
            print(f"❌ 本地加载失败 {tokenizer_path}: {e}")

    # 如果本地路径失败，尝试在线下载（但设置超时）
    print("🔄 尝试在线下载分词器...")
    try:
        import socket
        socket.setdefaulttimeout(30)  # 设置30秒超时

        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        print("✅ 在线下载分词器成功")

        # print(f"分词器词汇表大小: {len(tokenizer)}")
        # print(f"填充token ID: {tokenizer.pad_token_id}")

        # 自动保存到本地供下次使用
        os.makedirs("../data/tokenizer/facebook/bart-base", exist_ok=True)
        tokenizer.save_pretrained("../data/tokenizer/facebook/bart-base")
        print("💾 分词器已保存到本地缓存")

        return tokenizer

    except Exception as e:
        print(f"❌ 在线下载也失败: {e}")

def get_data_loaders(config):
    """
    创建训练和验证数据加载器
    这个函数负责完整的数据处理流程：
    1. 加载原始数据集
    2. 加载分词器
    3. 创建Dataset实例
    4. 创建DataLoader实例
    Args:
        config: 配置对象，包含以下属性：
            - dataset_name: 数据集名称（如"knkarthick/samsum"）
            - batch_size: 批次大小
            - max_input_length: 输入序列最大长度
            - max_target_length: 目标序列最大长度

    Returns:
        tuple: (train_loader, val_loader, tokenizer)
            - train_loader: 训练数据加载器
            - val_loader: 验证数据加载器
            - tokenizer: 分词器实例（后续用于模型和推理）
    """

    print("🚫 完全使用本地数据，避免网络连接...")

    # 1. 从本地加载数据集
    try:
        dataset = load_local_samsum_dataset(config.local_data_dir)
        print("✅ 成功从本地文件加载数据集")
    except Exception as e:
        print(f"❌ 从本地文件加载失败: {e}")

    # print(dataset)
    '''
    返回的dataset是一个DatasetDict对象，包含'train', 'validation', 'test'
    DatasetDict({
        train: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 14731
        })
        validation: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 818
        })
        test: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 819
        })
    })
    '''

    # 2. 从本地加载分词器
    tokenizer = load_local_tokenizer(config.local_tokenizer_path)
    # 分词器功能：
    # - 将文本转换为token ID序列
    # - 添加特殊token（如[CLS], [SEP], [PAD]等）
    # - 处理填充和截断

    # 3. 创建训练集Dataset实例
    train_dataset = SAMSumDataset(
        dataset['train'],  # 训练集数据
        tokenizer,  # 分词器
        config.max_input_length,  # 输入序列最大长度
        config.max_target_length  # 目标序列最大长度
    )
    # 此时train_dataset已经包含了所有预处理逻辑

    # 4. 创建验证集Dataset实例
    val_dataset = SAMSumDataset(
        dataset['validation'],  # 验证集数据
        tokenizer,  # 使用同一个分词器（保证词汇表一致）
        config.max_input_length,
        config.max_target_length
    )

    # 5. 创建测试集Dataset实例
    test_dataset = SAMSumDataset(
        dataset['test'],  # 测试集数据
        tokenizer,  # 使用同一个分词器（保证词汇表一致）
        config.max_input_length,
        config.max_target_length
    )

    # 6. 创建数据加载器（DataLoader）
    # 训练数据加载器 - 支持随机打乱
    train_loader = DataLoader(
        train_dataset,  # 训练数据集
        batch_size=config.batch_size,  # 每个批次的样本数
        shuffle=True,  # 每个epoch打乱数据顺序（防止模型记忆顺序）
        num_workers=4,# 数据加载的进程数（加速数据加载）
        pin_memory=True,# 是否将数据固定到GPU内存（加速GPU传输）
    )
    # 训练时打乱数据有助于：
    # - 防止模型过拟合于数据顺序
    # - 提高模型的泛化能力

    # 验证数据加载器 - 不需要打乱
    val_loader = DataLoader(
        val_dataset,  # 验证数据集
        batch_size=config.batch_size,  # 批次大小（通常与训练集相同）
        shuffle=False,  # 验证时不打乱数据（保证评估的一致性）
        num_workers=4,
        pin_memory=True,
    )
    # 验证时不打乱的原因：
    # - 保证每次验证的结果可比较
    # - 便于调试和结果分析

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 测试集绝对不打乱
        num_workers=4,
        pin_memory=True,
    )

    # 返回创建好的数据加载器和分词器
    return train_loader, val_loader,test_loader, tokenizer

'''后台开VPN才能在线下载数据集和分词器，如果不开请注释本函数，取消另一个get_data_loaders函数的注释'''
# def get_data_loaders(config):
#     """
#     创建训练和验证数据加载器
#     这个函数负责完整的数据处理流程：
#     1. 加载原始数据集
#     2. 加载分词器
#     3. 创建Dataset实例
#     4. 创建DataLoader实例
#     Args:
#         config: 配置对象，包含以下属性：
#             - dataset_name: 数据集名称（如"knkarthick/samsum"）
#             - batch_size: 批次大小
#             - max_input_length: 输入序列最大长度
#             - max_target_length: 目标序列最大长度
#
#     Returns:
#         tuple: (train_loader, val_loader, tokenizer)
#             - train_loader: 训练数据加载器
#             - val_loader: 验证数据加载器
#             - tokenizer: 分词器实例（后续用于模型和推理）
#     """
#
#     # 1. 加载数据集
#     # 使用Hugging Face的load_dataset函数从Hub下载或加载本地数据集
#     dataset = load_dataset(config.dataset_name)
#
#     # print(dataset)
#     '''
#     返回的dataset是一个DatasetDict对象，包含'train', 'validation', 'test'
#     DatasetDict({
#         train: Dataset({
#             features: ['id', 'dialogue', 'summary'],
#             num_rows: 14731
#         })
#         validation: Dataset({
#             features: ['id', 'dialogue', 'summary'],
#             num_rows: 818
#         })
#         test: Dataset({
#             features: ['id', 'dialogue', 'summary'],
#             num_rows: 819
#         })
#     })
#     '''
#
#
#     # 2. 加载分词器
#     # 使用BART模型的分词器，因为BART在摘要任务上表现优秀
#     tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
#     # 分词器功能：
#     # - 将文本转换为token ID序列
#     # - 添加特殊token（如[CLS], [SEP], [PAD]等）
#     # - 处理填充和截断
#
#     # 3. 创建训练集Dataset实例
#     train_dataset = SAMSumDataset(
#         dataset['train'],  # 训练集数据
#         tokenizer,  # 分词器
#         config.max_input_length,  # 输入序列最大长度
#         config.max_target_length  # 目标序列最大长度
#     )
#     # 此时train_dataset已经包含了所有预处理逻辑
#
#     # 4. 创建验证集Dataset实例
#     val_dataset = SAMSumDataset(
#         dataset['validation'],  # 验证集数据
#         tokenizer,  # 使用同一个分词器（保证词汇表一致）
#         config.max_input_length,
#         config.max_target_length
#     )
#
#     # 5. 创建测试集Dataset实例
#     test_dataset = SAMSumDataset(
#         dataset['test'],  # 测试集数据
#         tokenizer,  # 使用同一个分词器（保证词汇表一致）
#         config.max_input_length,
#         config.max_target_length
#     )
#
#     # 6. 创建数据加载器（DataLoader）
#     # 训练数据加载器 - 支持随机打乱
#     train_loader = DataLoader(
#         train_dataset,  # 训练数据集
#         batch_size=config.batch_size,  # 每个批次的样本数
#         shuffle=True,  # 每个epoch打乱数据顺序（防止模型记忆顺序）
#         num_workers=4,# 数据加载的进程数（加速数据加载）
#         pin_memory=True,# 是否将数据固定到GPU内存（加速GPU传输）
#     )
#     # 训练时打乱数据有助于：
#     # - 防止模型过拟合于数据顺序
#     # - 提高模型的泛化能力
#
#     # 验证数据加载器 - 不需要打乱
#     val_loader = DataLoader(
#         val_dataset,  # 验证数据集
#         batch_size=config.batch_size,  # 批次大小（通常与训练集相同）
#         shuffle=False,  # 验证时不打乱数据（保证评估的一致性）
#         num_workers=4,
#         pin_memory=True,
#     )
#     # 验证时不打乱的原因：
#     # - 保证每次验证的结果可比较
#     # - 便于调试和结果分析
#
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config.batch_size,
#         shuffle=False,  # 测试集绝对不打乱
#         num_workers=4,
#         pin_memory=True,
#     )
#
#     # 返回创建好的数据加载器和分词器
#     return train_loader, val_loader,test_loader, tokenizer

# train_loader, val_loader,test_loader, tokenizer = get_data_loaders(config)
