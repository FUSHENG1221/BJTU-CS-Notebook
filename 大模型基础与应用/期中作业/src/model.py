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

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        self.d_v = d_model // n_heads

        # 投影矩阵: W_i^Q, W_i^K, W_i^V ∈ ℝ^(d_model × d_k)
        self.W_Q = nn.Linear(d_model, d_model)  # 等价于所有W_i^Q的拼接
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影矩阵: W^O ∈ ℝ^(d_model × d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # 缩放点积注意力层
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        实现 MultiHead(Q, K, V) = Concat(head_1, ..., head_H)W^O
        其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        """
        batch_size, seq_len = Q.size(0), Q.size(1)

        # 1. 线性投影并分头: QW^Q, KW^K, VW^V
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 2. 每个头独立计算缩放点积注意力
        # 输出形状: (batch_size, n_heads, seq_len, d_v)
        output, attn_weights = self.attention(Q, K, V, mask)

        # 3. 拼接所有头: Concat(head_1, ..., head_H)
        # 形状: (batch_size, seq_len, n_heads * d_v) = (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v
        )

        # 4. 最终投影: Concat(head_1, ..., head_H)W^O
        output = self.W_O(output)

        return output, attn_weights


class PositionwiseFFN(nn.Module):
    """逐位置前馈网络 - 完全按照报告中的公式实现"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFFN, self).__init__()

        # 第一层线性变换: W1 ∈ ℝ^(d_model × d_ff), b1 ∈ ℝ^(d_ff)
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二层线性变换: W2 ∈ ℝ^(d_ff × d_model), b2 ∈ ℝ^(d_model)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 使用ReLU激活函数

    def forward(self, x):
        """
        实现 FFN(x) = ReLU(xW1 + b1)W2 + b2

        Args:
            x: 输入张量 ∈ ℝ^(batch_size, seq_len, d_model)

        Returns:
            输出张量 ∈ ℝ^(batch_size, seq_len, d_model)
        """
        # 第一层变换: xW1 + b1
        hidden = self.linear1(x)

        # ReLU激活函数
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # 第二层变换: hiddenW2 + b2
        output = self.linear2(hidden)

        return output


class ResidualConnection(nn.Module):
    """残差连接 + 层归一化模块"""

    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        实现 LayerNorm(x + Sublayer(x))

        Args:
            x: 子层输入
            sublayer_output: 子层输出

        Returns:
            归一化后的残差连接结果
        """
        # 残差连接: x + Dropout(Sublayer(x))
        residual_output = x + self.dropout(sublayer_output)

        # 层归一化: LayerNorm(residual_output)
        output = self.norm(residual_output)

        return output


class EncoderLayer(nn.Module):
    """Transformer编码器层 - 包含多头自注意力和前馈网络"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 多头自注意力子层
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络子层
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

        # 两个残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        """
        编码器层前向传播:
        1. 多头自注意力 + 残差连接 + 层归一化
        2. 前馈网络 + 残差连接 + 层归一化
        """
        # 子层1: 多头自注意力
        attn_output, attn_weights = self.self_attention(x, x, x, mask)

        # 残差连接1: LayerNorm(x + MultiHeadAttention(x))
        x = self.residual1(x, attn_output)

        # 子层2: 前馈网络
        ffn_output = self.feed_forward(x)

        # 残差连接2: LayerNorm(x + FFN(x))
        x = self.residual2(x, ffn_output)

        return x, attn_weights


class DecoderLayer(nn.Module):
    """Transformer解码器层 - 包含掩码多头注意力和编码器-解码器注意力"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # 掩码多头自注意力（防止看到未来信息）
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 编码器-解码器注意力
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)

        # 三个残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        解码器层前向传播:
        1. 掩码自注意力 + 残差连接 + 层归一化
        2. 编码器-解码器注意力 + 残差连接 + 层归一化
        3. 前馈网络 + 残差连接 + 层归一化
        """
        # 子层1: 掩码自注意力
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.residual1(x, self_attn_output)

        # 子层2: 编码器-解码器注意力
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.residual2(x, cross_attn_output)

        # 子层3: 前馈网络
        ffn_output = self.feed_forward(x)
        x = self.residual3(x, ffn_output)

        return x, self_attn_weights, cross_attn_weights


class Transformer(nn.Module):
    """完整的Transformer模型 - 修复版本"""

    def __init__(self, config, tokenizer=None):
        super(Transformer, self).__init__()
        self.config = config

        # 关键修复：保存tokenizer引用
        self.tokenizer = tokenizer

        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 输出投影层
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def _create_src_mask(self, src):
        """创建源序列掩码 - 修复：使用tokenizer的pad_token_id"""
        device = next(self.parameters()).device
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 1  # 默认值，如果tokenizer不存在

        mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask.to(device)

    def _create_tgt_mask(self, seq_len):
        """创建目标序列掩码"""
        device = next(self.parameters()).device
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(1)

    def encode(self, src, src_mask=None):
        """编码器前向传播"""
        if src_mask is None:
            src_mask = self._create_src_mask(src)

        x = self.embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        encoder_self_attentions = []
        for layer in self.encoder_layers:
            x, self_attn = layer(x, src_mask)
            encoder_self_attentions.append(self_attn)

        return x, encoder_self_attentions

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """解码器前向传播"""
        if tgt_mask is None:
            tgt_mask = self._create_tgt_mask(tgt.size(1))

        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        decoder_self_attentions = []
        decoder_cross_attentions = []
        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            decoder_self_attentions.append(self_attn)
            decoder_cross_attentions.append(cross_attn)

        logits = self.output_projection(x)
        return logits, decoder_self_attentions, decoder_cross_attentions

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """完整的前向传播"""
        encoder_output, encoder_self_attentions = self.encode(src, src_mask)
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
        生成方法 - 修复版本
        """
        self.eval()

        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            if src_mask is None:
                src_mask = self._create_src_mask(src)

            # 编码输入序列
            encoder_output, _ = self.encode(src, src_mask)

            # 关键修复：使用tokenizer的起始token
            if self.tokenizer is not None:
                start_token_id = self.tokenizer.bos_token_id or 0
                eos_token_id = self.tokenizer.eos_token_id or 2
            else:
                start_token_id = 0
                eos_token_id = 2

            # 初始化目标序列
            tgt = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

            # 生成序列
            for step in range(max_length - 1):
                tgt_mask = self._create_tgt_mask(tgt.size(1))

                # 解码一步
                logits, _, _ = self.decode(tgt, encoder_output, src_mask, tgt_mask)
                next_token_logits = logits[:, -1, :] / temperature

                # 贪心搜索
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 检查是否所有序列都生成了结束token
                if (next_token == eos_token_id).all():
                    break

            return tgt