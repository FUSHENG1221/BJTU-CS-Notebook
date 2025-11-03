import torch

class Config:
    # 数据集配置
    dataset_name = "knkarthick/samsum"
    max_input_length = 512
    max_target_length = 128

    # 模型架构配置
    vocab_size = 50265  # BART的词汇表大小
    d_model = 256  # 模型维度
    n_heads = 4  # 注意力头数
    n_layers = 2  # Encoder/Decoder层数
    d_ff = 512  # 前馈网络维度
    dropout = 0.1  # Dropout率
    max_seq_len = 512  # 最大序列长度

    # 训练配置
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 10
    warmup_steps = 1000
    max_grad_norm = 1.0

    # 实验配置
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")


config = Config()