import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask（如果有）
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        if mask is not None:
            # 确保mask在正确的设备上
            if mask.device != scores.device:
                mask = mask.to(scores.device)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 输出线性变换
        output = self.w_o(attn_output)
        return output, attn_weights

class PositionwiseFFN(nn.Module):
    """位置级前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, attn_weights

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力（掩码防止看到未来信息）
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力（关注编码器输出）
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x, self_attn_weights, cross_attn_weights

class Transformer(nn.Module):
    """完整的Encoder-Decoder Transformer模型"""
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        
        # 词嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def encode(self, src, src_mask=None):
        # 输入嵌入 + 位置编码
        x = self.embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        encoder_self_attentions = []
        for layer in self.encoder_layers:
            x, self_attn = layer(x, src_mask)
            encoder_self_attentions.append(self_attn)
        
        return x, encoder_self_attentions

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # 目标嵌入 + 位置编码
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过解码器层
        decoder_self_attentions = []
        decoder_cross_attentions = []
        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            decoder_self_attentions.append(self_attn)
            decoder_cross_attentions.append(cross_attn)
        
        # 输出投影
        logits = self.output_projection(x)
        return logits, decoder_self_attentions, decoder_cross_attentions

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器前向传播
        encoder_output, encoder_self_attentions = self.encode(src, src_mask)
        
        # 解码器前向传播
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
        """生成摘要"""
        self.eval()

        with torch.no_grad():
            # 编码输入
            encoder_output, _ = self.encode(src, src_mask)

            # 初始化目标序列（起始token）
            batch_size = src.size(0)
            tgt = torch.ones(batch_size, 1, dtype=torch.long, device=src.device) * 0  # 假设0是起始token

            for _ in range(max_length - 1):
                # 创建目标掩码（防止看到未来信息）
                tgt_mask = self._create_tgt_mask(tgt.size(1))

                # 解码一步
                logits, _, _ = self.decode(tgt, encoder_output, src_mask, tgt_mask)
                next_token_logits = logits[:, -1, :] / temperature

                # 采样下一个token
                next_token = torch.softmax(next_token_logits, dim=-1).multinomial(1)
                tgt = torch.cat([tgt, next_token], dim=1)

                # 如果生成了结束token，停止生成
                if (next_token == 1).all():  # 假设1是结束token
                    break

            return tgt

    def _create_src_mask(self, src):
        """创建源序列掩码（忽略填充token）"""
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def _create_tgt_mask(self, seq_len):
        """创建目标序列掩码（防止看到未来信息）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(1)