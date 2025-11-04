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


# def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
#     """
#     保存模型检查点:
#     用于在训练过程中保存模型的当前状态，包括：
#     - 模型参数
#     - 优化器状态
#     - 训练进度信息
#     Args:
#         model (torch.nn.Module): 要保存的模型
#         optimizer (torch.optim.Optimizer): 优化器
#         epoch (int): 当前训练轮数
#         loss (float): 当前损失值
#         filepath (str): 保存路径
#     """
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),  # 模型参数
#         'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
#         'loss': loss,
#         'model_config': model.config.__dict__ if hasattr(model, 'config') else {}
#     }
#
#     torch.save(checkpoint, filepath)
#     print(f"✅ 模型检查点已保存: {filepath}")
#
#
# def load_model_checkpoint(model, optimizer, filepath):
#     """
#     加载模型检查点:
#     用于从保存的检查点恢复训练状态
#     Args:
#         model (torch.nn.Modimizerule): 要加载参数的模型
#         optimizer (torch.optim.Optimizer): 要加载状态的优化器
#         filepath (str): 检查点文件路径
#     Returns:
#         tuple: (epoch, loss) - 恢复的训练轮数和损失值
#     """
#     checkpoint = torch.load(filepath)
#
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     print(f"✅ 模型检查点已加载: {filepath}")
#     print(f"   恢复训练轮数: {checkpoint['epoch']}")
#     print(f"   恢复损失值: {checkpoint['loss']:.4f}")
#
#     return checkpoint['epoch'], checkpoint['loss']
#
#
# def calculate_model_size(model):
#     """
#     计算模型在磁盘上的大致大小
#
#     Args:
#         model (torch.nn.Module): PyTorch模型
#
#     Returns:
#         dict: 包含不同单位下模型大小的字典
#     """
#     # 计算参数数量
#     num_params = count_parameters(model)
#
#     # 假设每个参数是32位浮点数（4字节）
#     size_bytes = num_params * 4
#
#     # 转换为不同单位
#     size_info = {
#         'parameters': num_params,
#         'bytes': size_bytes,
#         'kilobytes': size_bytes / 1024,
#         'megabytes': size_bytes / (1024 * 1024),
#         'gigabytes': size_bytes / (1024 * 1024 * 1024)
#     }
#
#     return size_info
#
#
# def print_model_summary(model, tokenizer=None):
#     """
#     打印模型的详细摘要信息
#
#     Args:
#         model (torch.nn.Module): 要分析的模型
#         tokenizer: 可选的分词器（用于显示词汇表信息）
#     """
#     print("=" * 60)
#     print("模型摘要信息")
#     print("=" * 60)
#
#     # 基本信息
#     num_params = count_parameters(model)
#     size_info = calculate_model_size(model)
#
#     print(f"模型类型: {model.__class__.__name__}")
#     print(f"可训练参数数量: {num_params:,}")
#     print(f"模型大小: {size_info['megabytes']:.2f} MB")
#
#     # 分层信息
#     print("\n模型层信息:")
#     for name, module in model.named_children():
#         num_layer_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
#         print(f"  {name}: {num_layer_params:,} 参数")
#
#     # 分词器信息（如果提供）
#     if tokenizer is not None:
#         print(f"\n分词器信息:")
#         print(f"  类型: {tokenizer.__class__.__name__}")
#         print(f"  词汇表大小: {tokenizer.vocab_size:,}")
#         print(f"  特殊token: {list(tokenizer.special_tokens_map.values())}")
#
#     print("=" * 60)
#
#
# def visualize_attention(attention_weights, tokens, layer_idx=0, head_idx=0):
#     """
#     可视化注意力权重（简化版）
#
#     Args:
#         attention_weights (list): 注意力权重列表
#         tokens (list): 对应的token列表
#         layer_idx (int): 要可视化的层索引
#         head_idx (int): 要可视化的头索引
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     # 获取指定层和头的注意力权重
#     # attention_weights[layer_idx] 形状: (batch_size, num_heads, seq_len, seq_len)
#     attn_matrix = attention_weights[layer_idx][0, head_idx].cpu().numpy()
#
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(attn_matrix,
#                 xticklabels=tokens,
#                 yticklabels=tokens,
#                 cmap='viridis',
#                 annot=True,  # 显示数值
#                 fmt='.2f',
#                 cbar_kws={'label': 'Attention Weight'})
#
#     plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
#     plt.xlabel('Key Tokens')
#     plt.ylabel('Query Tokens')
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show()
#
#
# # 使用示例
# if __name__ == "__main__":
#     # 示例用法
#     print("工具函数演示:")
#
#     # 1. 设置随机种子
#     set_seed(42)
#     print("✅ 随机种子设置完成")
#
#
#     # 2. 创建一个简单的模型进行测试
#     class SimpleModel(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.linear1 = torch.nn.Linear(10, 20)
#             self.linear2 = torch.nn.Linear(20, 5)
#
#         def forward(self, x):
#             return self.linear2(self.linear1(x))
#
#
#     model = SimpleModel()
#
#     # 3. 计算参数数量
#     num_params = count_parameters(model)
#     print(f"✅ 模型参数数量: {num_params}")
#
#     # 4. 计算模型大小
#     size_info = calculate_model_size(model)
#     print(f"✅ 模型大小: {size_info['kilobytes']:.2f} KB")
#
#     # 5. 打印模型摘要
#     print_model_summary(model)
#
#     print("所有工具函数测试完成! 🎉")