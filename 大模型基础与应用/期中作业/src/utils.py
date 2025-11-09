import torch
import numpy as np
import random


def set_seed(seed):
    """
    设置随机种子以确保实验的可重复性
    在机器学习中，随机种子用于控制随机数生成器的初始状态，确保每次运行代码时
    产生的随机结果是一致的。这对于实验的可重复性和调试非常重要。
    Args:
        seed (int): 随机种子值，通常使用42、1234等常用值
    """
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python内置random模块的随机种子

    if torch.cuda.is_available():
        # 如果CUDA可用，还需要设置GPU的随机种子
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """
    计算模型中可训练参数的数量
    这个函数用于分析模型的复杂度和大小，帮助了解：
    - 模型的计算需求
    - 内存占用估计
    - 训练时间预估
    Args:
        model (torch.nn.Module): PyTorch模型实例
    Returns:
        int: 模型中所有可训练参数的总数量
    """
    # 使用生成器表达式遍历所有参数，只计算requires_grad=True的参数
    # p.numel() 返回参数张量中的元素数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_attention_patterns(model, tokenizer, text):
    """
    分析Transformer模型的注意力模式
    这个函数用于可视化和理解模型在处理文本时的注意力分布，帮助：
    - 理解模型关注的重点
    - 调试注意力机制
    - 分析模型的可解释性
    Args:
        model (torch.nn.Module): Transformer模型实例
        tokenizer: 分词器，用于将文本转换为token
        text (str): 要分析的输入文本
    Returns:
        list: 包含所有编码器层注意力权重的列表
              每个元素是一个张量，形状为 (batch_size, num_heads, seq_len, seq_len)
    """
    # 将模型设置为评估模式
    # 这会禁用dropout和batch normalization的训练特定行为
    model.eval()

    # 使用torch.no_grad()上下文管理器禁用梯度计算
    # 这可以节省内存并加速推理过程
    with torch.no_grad():
        # 1. 使用分词器将文本转换为模型输入格式
        # return_tensors='pt' 表示返回PyTorch张量
        inputs = tokenizer(text, return_tensors='pt')

        # 提取输入token IDs
        # input_ids 形状: (batch_size, sequence_length)
        input_ids = inputs['input_ids']

        # 2. 前向传播获取模型输出
        # 这里使用输入同时作为源序列和目标序列（自编码任务）
        outputs = model(input_ids, input_ids)

        # 3. 提取注意力权重
        # encoder_self_attentions 包含所有编码器层的自注意力权重
        # 每个注意力权重的形状: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = outputs['encoder_self_attentions']

        return attention_weights