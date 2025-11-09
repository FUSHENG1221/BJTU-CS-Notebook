import os

import torch

class Config:
    # 数据集配置
    dataset_name = "knkarthick/samsum"
    max_input_length = 512# 输入序列的最大长度（对话文本）
    max_target_length = 128# 目标序列的最大长度（摘要文本）

    # 模型架构配置
    vocab_size = 50265  # 词汇表大小，对应BART-base分词器的词汇量
    d_model = 256  # 模型维度（隐藏层大小）:决定模型表示能力的核心参数
    n_heads = 4  # 必须能被d_model整除
    n_layers = 2  # Encoder/Decoder层数:原始Transformer使用6层，这里使用较少层数加快训练;层数越多，模型容量越大，但训练时间越长
    d_ff = 512  # 前馈网络（FFN）的隐藏层维度,通常是d_model的2-4倍
    dropout = 0.1  # Dropout率（丢弃率），正则化技术;随机丢弃一定比例的神经元，防止过拟合
    max_seq_len = 512  # 模型支持的最大序列长度,必须 >= max_input_length

    # 训练配置
    batch_size = 16 # 批次大小，每次训练使用的样本数量,较大的batch_size训练更稳定，但需要更多内存
    learning_rate = 3e-4# 学习率，控制参数更新步长;3e-4是Adam优化器的常用学习率
    # num_epochs = 10 # 训练轮数，整个数据集被完整训练的次数
    num_epochs = 20
    warmup_steps = 1000# 学习率预热步数,在训练初期逐步增加学习率，避免初始震荡,有助于训练稳定性
    max_grad_norm = 1.0# 最大梯度范数阈值：用于梯度裁剪，防止梯度爆炸

    # 实验配置
    seed = 42# 随机种子值，确保实验可重复;42是深度学习中的"经典"随机种子;固定种子可以确保每次运行结果一致
    device = "cuda" if torch.cuda.is_available() else "cpu"# 自动检测并使用GPU（如果可用），否则使用CPU

    @property
    def device(self):
        """
        设备属性获取器 - 返回适当的torch.device对象
        使用property装饰器创建计算属性，确保每次获取的都是
        最新的设备状态（特别是GPU可用性可能变化时）
        Returns:
            torch.device: CUDA设备（如果可用）或CPU设备
        """
        if torch.cuda.is_available():
            # 返回第一个GPU设备
            return torch.device("cuda:0")
        else:
            # 返回CPU设备
            return torch.device("cpu")

    # 模型保存配置
    save_dir = "..\\results\\model"  # 模型保存目录
    checkpoint_dir = "..\\results\\checkpoints"  # 检查点目录
    image_dir = "..\\results\\images"  # 图表保存目录
    local_tokenizer_path = "../data/tokenizer/facebook/bart-base"
    local_data_dir = "../data/samsum"

    def __init__(self):
        """初始化时创建必要的目录"""
        self.create_directories()

    def create_directories(self):
        """创建所有必要的目录"""
        directories = [self.save_dir, self.checkpoint_dir, self.image_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            # print(f"✅ 确保目录存在: {directory}")


config = Config()