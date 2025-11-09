import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Sinusoidal Positional Encoding）:
    Transformer架构的核心组件，用于向模型注入序列中每个token的位置信息。
    由于Transformer的自注意力机制本身不具备位置感知能力，
    需要额外添加位置编码来让模型理解序列中元素的顺序关系。

    基于论文《Attention Is All You Need》中的原始设计，
    使用不同频率的正弦和余弦函数为每个位置生成唯一的编码向量。

    Args:
        d_model (int): 词嵌入的维度（隐藏层大小），必须与Transformer模型的隐藏维度一致
        max_len (int): 支持的最大序列长度，默认5000个token
    """

    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码矩阵：
        预先计算所有可能位置的位置编码并缓存，避免在每次前向传播时重复计算。
        这种固定编码方式允许模型泛化到比训练时更长的序列。
        Args:
            d_model: 模型隐藏层维度，如256、512、768等
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵 PE ∈ ℝ^(max_len × d_model)
        # 初始化为全零，后续填充正弦余弦值
        pe = torch.zeros(max_len, d_model)

        # 位置索引 pos ∈ [0, max_len-1]
        # 创建位置序列：[0, 1, 2, ..., max_len-1]
        # unsqueeze(1) 将一维张量变为二维，便于后续的广播计算
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率项: div_term = 1 / 10000^(2i/d_model)
        # 使用指数和对数变换避免数值计算问题，提高数值稳定性
        # 数学等价于: div_term = 1 / (10000^(torch.arange(0, d_model, 2) / d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *  # 生成偶数索引: [0, 2, 4, ..., d_model-2]
            (-math.log(10000.0) / d_model)  # 计算缩放因子: -log(10000)/d_model
        )

        # 偶数维度使用正弦函数: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # 对矩阵的偶数列（索引0,2,4,...）应用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度使用余弦函数: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        # 对矩阵的奇数列（索引1,3,5,...）应用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 调整形状并注册为缓冲区
        pe = pe.unsqueeze(0).transpose(0, 1)  # 形状: (max_len, 1, d_model)
        # 注册为缓冲区（不参与梯度更新的持久化参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：将位置编码加到输入嵌入上
        将预计算的位置编码与输入序列的词嵌入相加，使模型同时获得词汇语义和位置信息。
        这种加法操作相当于给每个token的表示向量添加了位置特征。
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
                - batch_size: 批次大小
                - seq_len: 序列实际长度（必须 <= max_len）
                - d_model: 词嵌入维度（必须与初始化参数一致）
        Returns:
            Tensor: 添加位置编码后的张量，形状与输入x相同: (batch_size, seq_len, d_model)
        """
        # 添加位置编码: x = x + PE
        # 取前seq_len个位置编码，调整形状与x匹配后相加
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力（Scaled Dot-Product Attention）：
    Transformer架构的核心组件，用于计算查询（Query）与键（Key）之间的相似度，
    并基于该相似度对值（Value）进行加权求和。这是自注意力机制的基础操作。

    基于论文《Attention Is All You Need》中的原始设计，通过缩放点积来避免梯度消失问题，提高训练稳定性。

    Args:
        d_k (int): 键和查询向量的维度（每个注意力头的维度）
        dropout (float): Dropout率，用于防止过拟合，默认0.1
    """
    def __init__(self, d_k, dropout=0.1):
        """
        初始化缩放点积注意力层
        Args:
            d_k: 键/查询向量的维度，决定了缩放因子的大小
            dropout: 注意力权重的dropout率，提供正则化效果
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k  # 保存键/查询维度
        self.scale = math.sqrt(d_k)  # 缩放因子 √d_k，用于稳定梯度
        self.dropout = nn.Dropout(dropout)  # Dropout层，随机丢弃注意力权重

    def forward(self, Q, K, V, mask=None):
        """
        前向传播：计算缩放点积注意力
        实现公式：Attention(Q, K, V) = softmax(QK^T/√d_k)V
        Args:
            Q: 查询矩阵（Query），形状为 (batch_size, n_heads, L_Q, d_k)
                - batch_size: 批次大小
                - n_heads: 注意力头数
                - L_Q: 查询序列长度（目标序列长度）
                - d_k: 每个头的维度
            K: 键矩阵（Key），形状为 (batch_size, n_heads, L_K, d_k)
                - L_K: 键序列长度（源序列长度）
            V: 值矩阵（Value），形状为 (batch_size, n_heads, L_K, d_v)
                - d_v: 值向量的维度（通常等于d_k）
            mask: 注意力掩码，形状为 (batch_size, 1, L_Q, L_K) 或广播兼容的形状
                - 用于防止关注到填充位置或未来信息
                - 值为1的位置允许关注，值为0的位置被屏蔽
        Returns:
            tuple: (output, attn_weights)
                - output: 注意力输出，形状为 (batch_size, n_heads, L_Q, d_v)
                - attn_weights: 注意力权重，形状为 (batch_size, n_heads, L_Q, L_K)
        """
        # 计算注意力分数: scores = QK^T / √d_k
        # Q ∈ ℝ^(..., L_Q, d_k), K ∈ ℝ^(..., L_K, d_k) -> scores ∈ ℝ^(..., L_Q, L_K)
        # Q形状: (..., L_Q, d_k), K形状: (..., L_K, d_k)
        # 经过转置后K^T形状: (..., d_k, L_K)
        # 矩阵乘法结果形状: (..., L_Q, L_K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码（如果有）
        if mask is not None:
            # 确保掩码与分数矩阵在同一设备上（GPU/CPU）
            mask = mask.to(scores.device)
            # 将掩码中为0的位置（需要屏蔽的位置）填充为极小的负值
            # 这样在softmax计算中，这些位置的权重会趋近于0
            # 使用-1e9而不是-torch.inf是为了更好的数值稳定性
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重: attn_weights = softmax(scores)
        # 对最后一个维度（序列维度）应用softmax归一化
        # 将分数转换为概率分布，总和为1
        attn_weights = F.softmax(scores, dim=-1)
        # 应用dropout正则化：随机丢弃部分注意力权重
        # 防止模型过度依赖特定的注意力模式，提高泛化能力
        attn_weights = self.dropout(attn_weights)

        # 加权求和: output = attn_weights × V
        # attn_weights ∈ ℝ^(..., L_Q, L_K), V ∈ ℝ^(..., L_K, d_v) -> output ∈ ℝ^(..., L_Q, d_v)
        # 使用注意力权重对值向量V进行加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）：
    Transformer架构的核心组件，通过并行计算多个注意力头来增强模型的表示能力。
    每个头学习不同的表示子空间，最后将所有头的输出拼接并通过线性变换融合。

    基于论文《Attention Is All You Need》中的原始设计，公式为：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H)W^O
    其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        d_model (int): 模型的总维度（隐藏层大小）
        n_heads (int): 注意力头的数量
        dropout (float): Dropout率，默认0.1
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        初始化多头注意力层

        Args:
            d_model: 模型总维度，必须能被n_heads整除
            n_heads: 注意力头数量，决定并行计算的程度
            dropout: 注意力权重的dropout率
        """
        super(MultiHeadAttention, self).__init__()
        # 验证维度可除性
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model  # 总维度：如512
        self.n_heads = n_heads  # 头数量：如8
        self.d_k = d_model // n_heads  # 每个头的维度：如512÷8=64
        self.d_v = d_model // n_heads  # 值向量的维度（通常等于d_k）

        # 投影矩阵：将输入投影到不同的表示子空间
        # W_Q, W_K, W_V ∈ ℝ^(d_model × d_model)
        # 等价于将每个头的投影矩阵 W_i^Q, W_i^K, W_i^V ∈ ℝ^(d_model × d_k) 拼接而成
        self.W_Q = nn.Linear(d_model, d_model)  # 查询投影矩阵
        self.W_K = nn.Linear(d_model, d_model)  # 键投影矩阵
        self.W_V = nn.Linear(d_model, d_model)  # 值投影矩阵

        # 输出投影矩阵：W^O ∈ ℝ^(d_model × d_model)
        # 将拼接后的多头输出投影回原始维度
        self.W_O = nn.Linear(d_model, d_model)

        # 缩放点积注意力层（共享计算核心）
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)  # 输出dropout

    def forward(self, Q, K, V, mask=None):
        """
        前向传播：计算多头注意力
        实现多头注意力机制的四步流程：
            1. 线性投影并分头
            2. 并行计算缩放点积注意力
            3. 拼接所有头的输出
            4. 最终线性投影
        Args:
            Q: 查询矩阵，形状为 (batch_size, seq_len_q, d_model)
            K: 键矩阵，形状为 (batch_size, seq_len_kv, d_model)
            V: 值矩阵，形状为 (batch_size, seq_len_kv, d_model)
            mask: 注意力掩码，形状为 (batch_size, 1, seq_len_q, seq_len_kv)
        Returns:
            tuple: (output, attn_weights)
                - output: 多头注意力输出，形状为 (batch_size, seq_len_q, d_model)
                - attn_weights: 注意力权重，形状为 (batch_size, n_heads, seq_len_q, seq_len_kv)
        """
        batch_size, seq_len = Q.size(0), Q.size(1)

        # 1、对Q、K、V进行线性投影，然后分割成多个头
        # 原理：将高维空间投影到多个低维子空间，每个头关注不同特征

        # Q投影：QW^Q ∈ ℝ^(batch_size, seq_len_q, d_model)
        # 然后重塑为：ℝ^(batch_size, seq_len_q, n_heads, d_k)
        # 最后转置为：ℝ^(batch_size, n_heads, seq_len_q, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K投影：KW^K ∈ ℝ^(batch_size, seq_len_kv, d_model)
        # 重塑转置为：ℝ^(batch_size, n_heads, seq_len_kv, d_k)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V投影：VW^V ∈ ℝ^(batch_size, seq_len_kv, d_model)
        # 重塑转置为：ℝ^(batch_size, n_heads, seq_len_kv, d_v)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 2. 每个头独立计算缩放点积注意力
        # 输出形状: (batch_size, n_heads, seq_len_q, d_v)
        # attn_weights形状: (batch_size, n_heads, seq_len_q, seq_len_kv)
        output, attn_weights = self.attention(Q, K, V, mask)

        # 3. 拼接所有头: Concat(head_1, ..., head_H)
        # 将多个头的输出拼接回原始维度
        # 转置：output.transpose(1, 2).contiguous()
        # 转置：ℝ^(batch_size, seq_len_q, n_heads, d_v)
        # 重塑：ℝ^(batch_size, seq_len_q, n_heads * d_v) = ℝ^(batch_size, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v
        )

        # 将拼接后的输出投影到目标维度
        # Concat(head_1, ..., head_H)W^O ∈ ℝ^(batch_size, seq_len_q, d_model)
        output = self.W_O(output)
        output = self.dropout(output)  # 应用dropout正则化

        return output, attn_weights


class PositionwiseFFN(nn.Module):
    """
    逐位置前馈网络（Position-wise Feed-Forward Network）：
    Transformer架构中的关键组件，为每个位置（token）提供非线性变换能力。
    在自注意力机制之后应用，增强模型的表示能力和非线性拟合能力。
    基于论文《Attention Is All You Need》中的原始设计，公式为：
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    也称为"前馈层"或"全连接层"，但在Transformer中独立应用于每个位置。
    Args:
        d_model (int): 输入和输出的维度（隐藏层大小）
        d_ff (int): 中间隐藏层的维度，通常为d_model的2-4倍
        dropout (float): Dropout率，用于防止过拟合，默认0.1
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化逐位置前馈网络
        Args:
            d_model: 模型维度，决定了输入输出的大小
            d_ff: 前馈网络隐藏层维度，控制模型的容量和表达能力
            dropout: Dropout率，提供正则化效果防止过拟合
        """
        super(PositionwiseFFN, self).__init__()

        # 第一层线性变换: W1 ∈ ℝ^(d_model × d_ff), b1 ∈ ℝ^(d_ff)
        # 将输入从d_model维度投影到d_ff维度（升维）
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二层线性变换: W₂ ∈ ℝ^(d_ff × d_model), b₂ ∈ ℝ^(d_model)
        # 将隐藏层从d_ff维度投影回d_model维度（降维）
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)  # Dropout层，随机丢弃神经元
        self.activation = nn.ReLU()         # ReLU激活函数，提供非线性

    def forward(self, x):
        """
        前向传播：实现逐位置前馈变换
        实现 FFN(x) = ReLU(xW1 + b1)W2 + b2
        对输入序列中的每个位置独立应用相同的两层线性变换和激活函数。
        这是Transformer中主要的非线性变换来源。
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
                - batch_size: 批次大小
                - seq_len: 序列长度
                - d_model: 模型维度
        Returns:
            Tensor: 输出张量，形状与输入相同: (batch_size, seq_len, d_model)
        """
        # 第一层线性变换: hidden = xW₁ + b₁
        # 输入形状: (batch_size, seq_len, d_model)
        # 输出形状: (batch_size, seq_len, d_ff)
        hidden = self.linear1(x)

        # ReLU激活函数: hidden = max(0, hidden)
        # 引入非线性，增强模型表达能力
        # ReLU的选择基于原始论文，现代实现中也可使用GELU
        hidden = self.activation(hidden)
        # 应用Dropout: hidden = dropout(hidden)
        # 随机丢弃部分神经元，防止过拟合
        hidden = self.dropout(hidden)

        # 第二层线性变换: output = hiddenW₂ + b₂
        # 输入形状: (batch_size, seq_len, d_ff)
        # 输出形状: (batch_size, seq_len, d_model)
        output = self.linear2(hidden)

        return output


class ResidualConnection(nn.Module):
    """
    残差连接与层归一化模块（Residual Connection + Layer Normalization）：
    Transformer架构中的关键组件，用于解决深度神经网络中的梯度消失问题，
    并加速训练收敛。通过将输入直接添加到子层输出，确保信息能够跨层传播。
    实现公式：LayerNorm(x + Sublayer(x))
    基于论文《Attention Is All You Need》和深度残差网络（ResNet）的思想
    Args:
        d_model (int): 模型维度，决定层归一化的特征维度
        dropout (float): Dropout率，用于子层输出的随机丢弃，默认0.1
    """

    def __init__(self, d_model, dropout=0.1):
        """
        初始化残差连接模块
        Args:
            d_model: 模型维度，必须与输入张量的最后一个维度匹配
            dropout: Dropout率，控制子层输出的随机丢弃比例
        """
        super(ResidualConnection, self).__init__()
        # 层归一化（Layer Normalization）
        # 对每个样本的特征维度进行归一化，与Batch Normalization不同
        # 归一化维度：d_model（特征维度）
        self.norm = nn.LayerNorm(d_model)

        # Dropout层：随机丢弃子层输出的部分神经元
        # 提供正则化效果，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        前向传播：实现残差连接和层归一化
        计算流程：
            1. 将输入x与Dropout处理后的子层输出相加
            2. 对相加结果进行层归一化
            3. 返回归一化后的结果
        Args:
            x: 原始输入张量，形状为 (batch_size, seq_len, d_model)
                - 子层的输入，将直接添加到子层输出（恒等映射）
            sublayer_output: 子层输出张量，形状为 (batch_size, seq_len, d_model)
                - 来自自注意力层或前馈网络层的输出
                - 必须与输入x的形状完全相同
        Returns:
            Tensor: 归一化后的残差连接结果，形状与输入相同: (batch_size, seq_len, d_model)
        Raises:
            ValueError: 如果输入和子层输出的形状不匹配
        """
        # 确保输入和子层输出形状一致
        if x.shape != sublayer_output.shape:
            raise ValueError(f"输入形状{x.shape}与子层输出形状{sublayer_output.shape}不匹配")

        # 核心操作（残差连接）：x + Dropout(Sublayer(x))
        # 将原始输入直接添加到经过Dropout的子层输出中
        residual_output = x + self.dropout(sublayer_output)

        # 对残差连接的结果进行层归一化: LayerNorm(residual_output)
        # 归一化维度：最后一个维度（d_model）
        output = self.norm(residual_output)

        return output


class EncoderLayer(nn.Module):
    """
    Transformer编码器层（Encoder Layer）
    Transformer编码器的基本构建块，包含两个核心子层：
        1. 多头自注意力机制（Multi-Head Self-Attention）
        2. 逐位置前馈网络（Position-wise Feed-Forward Network）
    每个子层后都应用残差连接和层归一化，确保梯度有效传播和训练稳定性。
    这是Transformer架构处理输入序列的核心组件。
    Args:
        d_model (int): 模型维度（隐藏层大小）
        n_heads (int): 注意力头的数量
        d_ff (int): 前馈网络的隐藏层维度
        dropout (float): Dropout率，默认0.1
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        初始化编码器层
        Args:
            d_model: 模型维度，决定所有子层的输入输出维度
            n_heads: 注意力头数，必须能被d_model整除
            d_ff: 前馈网络隐藏层维度，通常为d_model的2-4倍
            dropout: Dropout率，用于防止过拟合
        """
        super(EncoderLayer, self).__init__()

        # 1、多头自注意力子层
        # 捕获输入序列中不同位置之间的依赖关系
        # 每个注意力头学习不同的表示模式
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 2、前馈网络子层
        # 为每个位置提供非线性变换能力，增强模型的表示能力和拟合能力
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

        # 3、残差连接模块
        # 每个子层后都应用残差连接和层归一化
        # 确保梯度有效传播，防止梯度消失
        self.residual1 = ResidualConnection(d_model, dropout)  # 自注意力后的残差连接
        self.residual2 = ResidualConnection(d_model, dropout)  # 前馈网络后的残差连接

    def forward(self, x, mask=None):
        """
        编码器层前向传播
        实现Transformer编码器的两阶段处理流程：
            1. 自注意力阶段：多头自注意力 + 残差连接 + 层归一化，捕获序列内部依赖关系
            2. 前馈网络阶段：前馈网络 + 残差连接 + 层归一化，增强每个位置的表示能力
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
                - batch_size: 批次大小
                - seq_len: 序列长度
                - d_model: 模型维度
            mask: 注意力掩码，形状为 (batch_size, 1, seq_len, seq_len)
                - 用于防止关注到填充位置（如[PAD] token）
                - None表示不使用掩码
        Returns:
            tuple: (output, attn_weights)
                - output: 编码器层输出，形状为 (batch_size, seq_len, d_model)
                - attn_weights: 注意力权重，形状为 (batch_size, n_heads, seq_len, seq_len)
        """
        # 子层1: 多头自注意力
        # 计算自注意力：让每个位置关注序列中的所有位置
        # 输入：x作为Q、K、V（自注意力）
        # 输出：attn_output包含经过注意力加权的信息
        attn_output, attn_weights = self.self_attention(x, x, x, mask)

        # 应用第一个残差连接：LayerNorm(x + Dropout(MultiHeadAttention(x)))
        # 保留原始输入信息，同时融入注意力加权信息
        x = self.residual1(x, attn_output)

        # 子层2: 前馈网络
        # 对每个位置独立应用非线性变换
        # 增强每个token的表示能力，不涉及位置间交互
        ffn_output = self.feed_forward(x)

        # 应用第二个残差连接：LayerNorm(x + Dropout(FFN(x)))
        # 保留自注意力后的信息，同时融入前馈网络的变换
        x = self.residual2(x, ffn_output)

        return x, attn_weights


class DecoderLayer(nn.Module):
    """
    Transformer解码器层（Decoder Layer）
    Transformer解码器的核心组件，包含三个关键子层：
        1. 掩码多头自注意力（Masked Multi-Head Self-Attention）
        2. 编码器-解码器注意力（Encoder-Decoder Attention）
        3. 逐位置前馈网络（Position-wise Feed-Forward Network）
    解码器层负责根据编码器输出和已生成的部分序列，预测下一个token。
    这是序列生成任务（如机器翻译、文本摘要）的核心组件。
    Args:
        d_model (int): 模型维度（隐藏层大小）
        n_heads (int): 注意力头的数量
        d_ff (int): 前馈网络的隐藏层维度
        dropout (float): Dropout率，默认0.1
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        初始化解码器层
        Args:
            d_model: 模型维度，决定所有子层的输入输出维度
            n_heads: 注意力头数，必须能被d_model整除
            d_ff: 前馈网络隐藏层维度，通常为d_model的2-4倍
            dropout: Dropout率，用于防止过拟合
        """
        super(DecoderLayer, self).__init__()

        # 1、掩码多头自注意力（防止看到未来信息）
        # 关注已生成的目标序列部分（防止信息泄露）
        # 使用因果掩码确保只能看到当前位置及之前的信息
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 2、编码器-解码器注意力子层
        # 关注编码器输出的源序列信息
        # 将源序列信息与目标序列对齐（类似传统机器翻译中的对齐机制）
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 3、前馈网络子层
        # 与编码器相同，提供非线性变换能力
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

        # 4、残差连接模块
        # 每个子层后都应用残差连接和层归一化
        self.residual1 = ResidualConnection(d_model, dropout)  # 自注意力后的残差连接
        self.residual2 = ResidualConnection(d_model, dropout)  # 交叉注意力后的残差连接
        self.residual3 = ResidualConnection(d_model, dropout)  # 前馈网络后的残差连接

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        解码器层前向传播
        实现Transformer解码器的三阶段处理流程：
            1. 掩码自注意力 + 残差连接 + 层归一化：关注已生成的目标序列
            2. 编码器-解码器注意力 + 残差连接 + 层归一化：将目标序列与源序列对齐
            3. 前馈网络 + 残差连接 + 层归一化：增强每个位置的表示能力
        Args:
            x: 目标序列输入（已生成的部分），形状为 (batch_size, tgt_seq_len, d_model)
                - 训练时：完整的target序列（右移一位）
                - 推理时：逐步生成的序列
            encoder_output: 编码器输出，形状为 (batch_size, src_seq_len, d_model)
                - 包含源序列的编码信息
            src_mask: 源序列掩码，形状为 (batch_size, 1, 1, src_seq_len)
                - 用于防止关注源序列中的填充token
            tgt_mask: 目标序列掩码，形状为 (batch_size, 1, tgt_seq_len, tgt_seq_len)
                - 因果掩码（下三角矩阵），防止看到未来信息
        Returns:
            tuple: (output, self_attn_weights, cross_attn_weights)
                - output: 解码器层输出，形状为 (batch_size, tgt_seq_len, d_model)
                - self_attn_weights: 自注意力权重，形状为 (batch_size, n_heads, tgt_seq_len, tgt_seq_len)
                - cross_attn_weights: 交叉注意力权重，形状为 (batch_size, n_heads, tgt_seq_len, src_seq_len)
        """
        # 子层1: 掩码自注意力
        # 关注目标序列中已生成的部分（防止信息泄露）
        # Q=K=V=x（自注意力），使用tgt_mask确保因果性
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        # 应用第一个残差连接：LayerNorm(x + Dropout(SelfAttention(x)))
        x = self.residual1(x, self_attn_output)

        # 子层2: 编码器-解码器注意力
        # 将目标序列与源序列对齐（类似传统对齐模型）
        # Q=x（目标序列），K=V=encoder_output（源序列）
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x,  # Query: 目标序列
            encoder_output,  # Key: 源序列
            encoder_output,  # Value: 源序列
            src_mask  # 源序列掩码（忽略填充token）
        )
        # 应用第二个残差连接：LayerNorm(x + Dropout(CrossAttention(x)))
        x = self.residual2(x, cross_attn_output)

        # 子层3: 前馈网络
        # 与编码器相同，增强每个位置的表示能力
        ffn_output = self.feed_forward(x)
        # 应用第三个残差连接：LayerNorm(x + Dropout(FFN(x)))
        x = self.residual3(x, ffn_output)

        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """
    完整的Transformer模型（Encoder-Decoder架构）
    基于论文《Attention Is All You Need》的完整实现，包含：
        1. 编码器（Encoder）：处理输入序列，提取特征表示
        2. 解码器（Decoder）：基于编码器输出生成目标序列
        3. 位置编码（PositionalEncoding）：注入序列位置信息
        4. 掩码机制：防止信息泄露和关注填充位置
    适用于序列到序列（Seq2Seq）任务，如：
        - 机器翻译
        - 文本摘要【本次实验的数据集】
        - 对话生成
        - 代码生成
    Args:
        config (object): 配置对象，包含模型超参数
        tokenizer (optional): 分词器实例，用于掩码生成和特殊token处理
    """

    def __init__(self, config, tokenizer=None):
        """
        初始化完整的Transformer模型
        Args:
            config: 配置对象，必须包含以下属性：
                - vocab_size: 词汇表大小
                - d_model: 模型维度
                - n_heads: 注意力头数
                - n_layers: 编码器/解码器层数
                - d_ff: 前馈网络隐藏层维度
                - dropout: Dropout率
                - max_seq_len: 最大序列长度
            tokenizer: 可选的分词器，用于掩码生成和特殊token处理
        """
        super(Transformer, self).__init__()
        self.config = config

        # 保存tokenizer引用：用于掩码生成和特殊token识别
        self.tokenizer = tokenizer

        # 词嵌入层
        # 将离散的token ID转换为连续的向量表示
        # 输入: token IDs ∈ [0, vocab_size-1]
        # 输出: 词向量 ∈ ℝ^(batch_size, seq_len, d_model)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 位置编码
        # 为序列添加位置信息，弥补自注意力机制的位置不敏感性
        # 使用正弦/余弦函数生成固定的位置编码
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

        # 编码器层，多层编码器堆叠，每层包含：
        # - 多头自注意力
        # - 前馈网络
        # - 残差连接和层归一化
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 解码器层，多层解码器堆叠，每层包含：
        # - 掩码多头自注意力（防止信息泄露）
        # - 编码器-解码器注意力（交叉注意力）
        # - 前馈网络
        # - 残差连接和层归一化
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 输出投影层
        # 将解码器输出投影回词汇表空间
        # 输入: ∈ ℝ^(batch_size, seq_len, d_model)
        # 输出: ∈ ℝ^(batch_size, seq_len, vocab_size)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)# Dropout层，用于正则化

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        权重初始化 - Xavier/Glorot初始化：
        确保模型训练开始时所有参数处于合适的数值范围，避免梯度爆炸或消失问题。
        """
        if isinstance(module, nn.Linear):
            # 线性层：Xavier均匀初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            # 嵌入层：正态分布初始化
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def _create_src_mask(self, src):
        """
        创建源序列掩码 - 屏蔽填充token：
        防止模型关注到源序列中的填充token（如[PAD]），这些token不包含有用信息。
        Args:
            src: 源序列token IDs，形状为 (batch_size, src_seq_len)
        Returns:
            mask: 源序列掩码，形状为 (batch_size, 1, 1, src_seq_len)
                  1表示有效位置，0表示填充位置（需要屏蔽）
        """
        device = next(self.parameters()).device
        # 获取填充token ID（优先使用tokenizer，否则使用默认值）
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 1  # 默认值，如果tokenizer不存在

        # 创建掩码：非填充位置为True，填充位置为False
        mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask.to(device)

    def _create_tgt_mask(self, seq_len):
        """
        创建目标序列掩码 - 因果掩码：
        防止解码器在生成时看到未来信息，确保自回归性质。使用下三角矩阵实现因果约束。
        Args:
            seq_len: 目标序列长度
        Returns:
            mask: 因果掩码，形状为 (1, 1, seq_len, seq_len)
                  下三角部分为1（允许关注），上三角部分为0（屏蔽）
        """
        device = next(self.parameters()).device
        # 创建下三角矩阵（因果掩码）
        # 例如 seq_len=4:
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(1)# 增加批次和头维度

    def encode(self, src, src_mask=None):
        """
        编码器前向传播 - 处理输入序列：
        将输入序列转换为富含语义的隐藏表示，捕获序列内部的长距离依赖关系。
        Args:
            src: 源序列token IDs，形状为 (batch_size, src_seq_len)
            src_mask: 源序列掩码，形状为 (batch_size, 1, 1, src_seq_len)
        Returns:
            tuple: (encoder_output, encoder_self_attentions)
                - encoder_output: 编码器输出，形状为 (batch_size, src_seq_len, d_model)
                - encoder_self_attentions: 各层自注意力权重列表
        """
        # 如果没有提供掩码，自动创建
        if src_mask is None:
            src_mask = self._create_src_mask(src)

        # 1. 词嵌入：token ID → 连续向量
        x = self.embedding(src)
        # 2. 位置编码：添加位置信息
        x = self.pos_encoding(x)
        # 3. 应用Dropout
        x = self.dropout(x)

        # 存储各层的注意力权重（用于可视化或分析）
        encoder_self_attentions = []
        # 4. 逐层处理：通过多个编码器层
        for layer in self.encoder_layers:
            x, self_attn = layer(x, src_mask)
            encoder_self_attentions.append(self_attn)

        return x, encoder_self_attentions

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        解码器前向传播 - 生成目标序列：
        基于编码器输出和已生成的目标序列部分，预测下一个token的概率分布。
        Args:
            tgt: 目标序列token IDs（通常右移一位），形状为 (batch_size, tgt_seq_len)
            encoder_output: 编码器输出，形状为 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码，形状为 (batch_size, 1, 1, src_seq_len)
            tgt_mask: 目标序列掩码，形状为 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        Returns:
            tuple: (logits, decoder_self_attentions, decoder_cross_attentions)
                - logits: 未归一化的输出分数，形状为 (batch_size, tgt_seq_len, vocab_size)
                - decoder_self_attentions: 解码器自注意力权重列表
                - decoder_cross_attentions: 解码器交叉注意力权重列表
        """
        # 如果没有提供掩码，自动创建
        if tgt_mask is None:
            tgt_mask = self._create_tgt_mask(tgt.size(1))

        # 1. 词嵌入：目标序列token ID → 连续向量
        x = self.embedding(tgt)
        # 2. 位置编码：添加位置信息
        x = self.pos_encoding(x)
        # 3. 应用Dropout
        x = self.dropout(x)

        # 存储各层的注意力权重：自注意力权重，交叉注意力权重
        decoder_self_attentions = []
        decoder_cross_attentions = []
        # 4. 逐层处理：通过多个解码器层
        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            decoder_self_attentions.append(self_attn)
            decoder_cross_attentions.append(cross_attn)

        # 5. 输出投影：隐藏表示 → 词汇表分数
        logits = self.output_projection(x)
        return logits, decoder_self_attentions, decoder_cross_attentions

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        完整的前向传播 - 训练模式：
        执行完整的编码器-解码器流程，用于模型训练。
        Args:
            src: 源序列token IDs，形状为 (batch_size, src_seq_len)
            tgt: 目标序列token IDs（通常右移一位），形状为 (batch_size, tgt_seq_len)
            src_mask: 源序列掩码（可选）
            tgt_mask: 目标序列掩码（可选）
        Returns:
            dict: 包含以下键的字典：
                - logits: 模型输出分数，形状为 (batch_size, tgt_seq_len, vocab_size)
                - encoder_self_attentions: 编码器各层自注意力权重
                - decoder_self_attentions: 解码器各层自注意力权重
                - decoder_cross_attentions: 解码器各层交叉注意力权重
        """
        # 1. 编码阶段：处理源序列
        encoder_output, encoder_self_attentions = self.encode(src, src_mask)
        # 2. 解码阶段：生成目标序列
        logits, decoder_self_attentions, decoder_cross_attentions = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )

        return {
            'logits': logits,
            'encoder_self_attentions': encoder_self_attentions,
            'decoder_self_attentions': decoder_self_attentions,
            'decoder_cross_attentions': decoder_cross_attentions
        }

    def generate(self, src, src_mask=None, max_length=128, temperature=1.0):
        """
        生成方法 - 推理模式（自回归生成）：
        使用贪心搜索或采样方法逐步生成目标序列，适用于推理/预测阶段。
        Args:
            src: 源序列token IDs，形状为 (batch_size, src_seq_len)
            src_mask: 源序列掩码（可选）
            max_length: 最大生成长度
            temperature: 温度参数，控制生成的随机性
                        >1.0：更随机，<1.0：更确定
        Returns:
            tgt: 生成的目标序列，形状为 (batch_size, generated_len)
        """
        # 设置为评估模式（禁用dropout等）
        self.eval()

        # 禁用梯度计算，节省内存
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # 如果没有提供掩码，自动创建
            if src_mask is None:
                src_mask = self._create_src_mask(src)

            # 1. 编码输入序列
            encoder_output, _ = self.encode(src, src_mask)

            # 2. 获取特殊token ID（用于开始和结束生成）
            if self.tokenizer is not None:
                start_token_id = self.tokenizer.bos_token_id or 0  # 起始token
                eos_token_id = self.tokenizer.eos_token_id or 2  # 结束token
            else:
                start_token_id = 0  # 默认起始token ID
                eos_token_id = 2  # 默认结束token ID

            # 3. 初始化目标序列（以起始token开始）
            tgt = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

            # 4. 自回归生成：逐个token生成
            for step in range(max_length - 1):
                # 创建当前序列的因果掩码
                tgt_mask = self._create_tgt_mask(tgt.size(1))

                # 解码一步：获取下一个token的logits
                logits, _, _ = self.decode(tgt, encoder_output, src_mask, tgt_mask)
                next_token_logits = logits[:, -1, :] / temperature  # 取最后一个位置的输出

                # 贪心搜索：选择概率最高的token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 将新token添加到序列中
                tgt = torch.cat([tgt, next_token], dim=1)

                # 检查是否所有序列都生成了结束token
                if (next_token == eos_token_id).all():
                    break

            return tgt