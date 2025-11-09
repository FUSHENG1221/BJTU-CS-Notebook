import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from datetime import datetime
from rouge_score import rouge_scorer
import math

from model import Transformer
from config import config
from data_loader import get_data_loaders
from utils import set_seed

import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class AblationTransformer(Transformer):
    """
    消融实验专用Transformer模型：支持启用/禁用位置编码功能
    这个类继承自基础的Transformer模型，通过控制位置编码的使用与否
    来研究位置信息在对话摘要任务中的重要性。
    消融实验(Ablation Study)是机器学习中常用的分析方法，通过
    移除模型的某个组件来评估该组件对整体性能的贡献。
    位置编码(Positional Encoding)是Transformer架构的关键组件，
    为模型提供序列中token的位置信息，因为自注意力机制本身是位置无关的。
    """
    def __init__(self, config, tokenizer=None, use_positional_encoding=True):
        """
        初始化消融实验模型
        Args:
            config: 模型配置对象，包含所有超参数设置
            tokenizer: 分词器实例，用于文本编码和解码
                - 提供词汇表大小、特殊token ID等信息
            use_positional_encoding: 是否使用位置编码（消融变量）
                - True: 使用标准位置编码（基线模型）
                - False: 禁用位置编码（消融模型）
        """
        # 调用父类构造函数，初始化基础Transformer架构
        super().__init__(config, tokenizer)
        # 存储消融实验配置
        self.use_positional_encoding = use_positional_encoding

        # 如果禁用位置编码，将位置编码层替换为恒等映射
        if not use_positional_encoding:
            # nn.Identity()是一个恒等映射层，输入什么就输出什么
            # 这相当于移除了位置信息，只保留词嵌入信息
            self.pos_encoding = nn.Identity()  # 恒等映射，不添加位置信息
            # 原始位置编码：x = word_embedding + positional_encoding
            # 消融版本：x = word_embedding + Identity(word_embedding) = word_embedding
            # 这样确保了输入维度的一致性，但移除了位置信息

    def encode(self, src, src_mask=None):
        """
        重写编码方法，支持位置编码开关
        编码器负责处理输入序列（对话文本），提取语义表示。
        这个方法的修改允许我们控制是否添加位置信息。
        Args:
            src: 源序列token IDs，形状为(batch_size, src_seq_len)
                包含对话文本的tokenized表示
            src_mask: 源序列掩码，形状为(batch_size, 1, 1, src_seq_len)
                用于标识哪些位置是填充token（需要被忽略）
        Returns:
            tuple: (encoder_output, encoder_self_attentions)
                - encoder_output: 编码器输出，形状为(batch_size, src_seq_len, d_model)
                  包含输入序列的上下文感知表示
                - encoder_self_attentions: 各层自注意力权重列表
                  用于可视化和分析模型关注点
        """
        # 如果没有提供掩码，自动创建（忽略填充token）
        if src_mask is None:
            src_mask = self._create_src_mask(src)

        # 1. 词嵌入：将离散的token ID转换为连续的向量表示
        # 形状: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(src)

        # 2. 条件位置编码：根据消融实验设置决定是否添加位置信息
        # 这是消融实验的核心修改点
        if self.use_positional_encoding:
            # 使用位置编码：添加正弦/余弦位置信息
            # 让模型能够感知token在序列中的位置
            x = self.pos_encoding(x)
        # else:
        #   不使用位置编码：保持原始的词语义嵌入
        #   模型只能基于内容信息，无法感知顺序关系

        # 3. 应用dropout进行正则化，防止过拟合
        x = self.dropout(x)

        # 4. 通过多层编码器进行处理
        encoder_self_attentions = []
        for layer in self.encoder_layers:
            # 每层包含：自注意力 + 前馈网络 + 残差连接 + 层归一化
            x, self_attn = layer(x, src_mask)
            # 存储注意力权重用于分析
            encoder_self_attentions.append(self_attn)

        return x, encoder_self_attentions

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        重写解码方法，支持位置编码开关
        解码器基于编码器输出和已生成的部分序列，预测下一个token。
        同样需要控制位置信息的使用。
        Args:
            tgt: 目标序列token IDs（已生成的部分），形状为(batch_size, tgt_seq_len)
                通常比源序列短，包含摘要文本的tokenized表示
            encoder_output: 编码器输出，形状为(batch_size, src_seq_len, d_model)
                包含输入序列的语义信息
            src_mask: 源序列掩码，形状为(batch_size, 1, 1, src_seq_len)
                防止关注到填充token
            tgt_mask: 目标序列掩码，形状为(batch_size, 1, tgt_seq_len, tgt_seq_len)
                因果掩码，防止看到未来信息（确保自回归性质）
        Returns:
            tuple: (logits, decoder_self_attentions, decoder_cross_attentions)
                - logits: 模型输出分数，形状为(batch_size, tgt_seq_len, vocab_size)
                  每个位置对词汇表中所有token的未归一化分数
                - decoder_self_attentions: 解码器自注意力权重列表
                  显示解码器如何关注已生成的目标序列部分
                - decoder_cross_attentions: 解码器-编码器交叉注意力权重列表
                  显示解码器如何关注源序列的不同部分（类似对齐机制）
        """
        # 如果没有提供目标掩码，自动创建因果掩码
        if tgt_mask is None:
            tgt_mask = self._create_tgt_mask(tgt.size(1))

        # 1. 词嵌入：目标序列的连续向量表示
        # 形状: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(tgt)

        # 2. 条件位置编码：同样根据消融设置决定
        if self.use_positional_encoding:
            # 使用位置编码：让解码器感知生成顺序
            x = self.pos_encoding(x)
        # else:
        #   不使用位置编码：解码器只能基于内容生成，无法感知生成顺序

        # 3. 应用dropout
        x = self.dropout(x)

        # 4. 通过多层解码器进行处理
        decoder_self_attentions = []  # 存储每层的自注意力权重
        decoder_cross_attentions = []  # 存储每层的交叉注意力权重
        for layer in self.decoder_layers:
            # 每层包含：掩码自注意力 + 编码器-解码器注意力 + 前馈网络 + 残差连接 + 层归一化
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            decoder_self_attentions.append(self_attn)
            decoder_cross_attentions.append(cross_attn)

        # 5. 输出投影：将隐藏表示映射回词汇表空间
        # 形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        logits = self.output_projection(x)

        return logits, decoder_self_attentions, decoder_cross_attentions


class AblationStudy:
    """
    位置编码消融实验类
    比较使用位置编码 vs 不使用位置编码的模型性能差异
    消融实验(Ablation Study)是机器学习中重要的分析方法，通过
    移除模型的某个组件来评估该组件对整体性能的贡献。
    本实验专门研究位置编码(Positional Encoding)在Transformer
    对话摘要任务中的重要性。
    实验设计：
    - 基线模型：使用标准位置编码（完整Transformer）
    - 消融模型：移除位置编码功能（使用恒等映射代替）
    - 对比指标：损失、困惑度、准确率、ROUGE分数
    """
    def __init__(self, config, tokenizer, train_loader, val_loader, test_loader):
        """
        初始化消融实验
        Args:
            config: 实验配置对象，包含所有超参数设置
                - device: 计算设备 (cuda/cpu)
                - num_epochs: 训练轮数
                - batch_size: 批次大小
                - learning_rate: 学习率
                - max_grad_norm: 梯度裁剪阈值
                - max_target_length: 目标序列最大长度
                - n_heads: 注意力头数量
                - d_model: 模型维度
            tokenizer: 分词器实例，用于文本编码和解码
                - vocab_size: 词汇表大小
                - pad_token_id: 填充token ID
                - 提供encode/decode功能
            train_loader: 训练数据加载器，包含训练数据集
                - 用于模型参数更新
                - 批次数据格式：{'input_ids': ..., 'attention_mask': ..., 'labels': ...}
            val_loader: 验证数据加载器，包含验证数据集
                - 用于模型性能评估和选择
                - 结构与训练加载器相同
            test_loader: 测试数据加载器，包含测试数据集
                - 用于最终性能评估（本实验未使用）
                - 结构与训练加载器相同
        实验设置说明：
        - 保持所有其他超参数相同，只改变位置编码的使用
        - 使用相同的随机种子确保可重复性
        - 在相同的硬件环境下运行两个实验
        """
        self.config = config
        self.config.num_epochs = 10# 为了节省时间，将训练轮数限制为10
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = config.device  # 计算设备（GPU/CPU）

        # 创建结果目录
        self.results_dir = "../results/ablation_study"
        os.makedirs(self.results_dir, exist_ok=True)
        # 目录结构：
        # ../results/ablation_study/
        #   ├── ablation_results_10epochs_8heads.json    # JSON格式完整结果
        #   ├── experiment_summary_10epochs_8heads.txt    # 文本摘要
        #   └── ablation_comparison_10epochs_8heads.png  # 可视化图表

        # 实验记录 - 结构化存储所有实验结果
        self.results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),  # 实验时间戳（ISO格式）
                "config": config.__dict__,  # 配置参数（转换为字典便于序列化）
                "description": "位置编码消融实验：比较使用位置编码 vs 不使用位置编码的模型性能"
            },
            "with_positional_encoding": {},  # 基线模型结果（有位置编码）
            "without_positional_encoding": {}  # 消融模型结果（无位置编码）
        }

    def calculate_perplexity(self, loss):
        """
        计算困惑度(Perplexity)
        困惑度是语言模型中最重要的评估指标之一，表示模型对测试数据的"困惑程度"。
        数学公式：perplexity = exp(loss)
        Args:
            loss: 交叉熵损失值
                - 类型: float
                - 范围: 0到正无穷（越小越好）
                - 来源: 交叉熵损失函数计算结果
        Returns:
            float: 困惑度值
                - 范围: 1到正无穷（越小越好）
                - 完美预测: 困惑度 = 1（损失为0时）
                - 随机猜测: 困惑度 ≈ 词汇表大小（最差情况）
        解释：
        - 困惑度可以理解为"模型需要平均考虑多少个候选词"
        - 值越小表示模型对数据的预测越确定和准确
        - 在语言模型中，困惑度通常比损失函数更直观
        """
        return math.exp(loss)

    def calculate_accuracy(self, logits, labels, ignore_index=-100):
        """
        计算准确率(Accuracy)
        准确率衡量模型预测正确的token比例，是直观的评估指标。
        只计算非填充位置的预测准确性。
        Args:
            logits: 模型输出，形状为(batch_size, seq_len, vocab_size)
                - 每个位置对词汇表中所有token的未归一化分数
                - 通过softmax可以转换为概率分布
            labels: 真实标签，形状为(batch_size, seq_len)
                - 目标序列的token ID
                - 包含特殊token（如填充token、起始符、结束符）
            ignore_index: 要忽略的标签索引
                - 默认值: -100（PyTorch标准忽略索引）
                - 实际使用: self.tokenizer.pad_token_id（填充token）
                - 作用: 避免填充位置影响准确率计算
        Returns:
            float: 准确率（0.0到1.0之间）
                - 0.0: 所有预测都错误
                - 0.5: 一半预测正确
                - 1.0: 所有预测都正确
        计算过程：
        1. 获取预测结果（概率最大的token）
        2. 创建有效token掩码（忽略填充位置）
        3. 计算正确预测的数量
        4. 计算有效token的总数
        5. 计算准确率 = 正确数量 / 有效总数
        """
        # 1. 获取预测结果 - 选择概率最大的token
        # torch.argmax返回每个位置最大值的索引
        # dim=-1表示在最后一个维度（vocab_size）上取最大值
        predictions = torch.argmax(logits, dim=-1)
        # 2. 创建有效token的掩码 - 标识哪些位置需要参与计算
        # 忽略填充token和其他指定忽略的token
        valid_mask = (labels != ignore_index)
        # 3. 计算正确预测的数量
        # 条件1: 预测等于真实标签
        # 条件2: 位置是有效的（非填充）
        correct = (predictions == labels) & valid_mask
        correct_count = correct.sum().item()  # 统计True的数量
        # 4. 计算有效token的总数
        total_valid = valid_mask.sum().item()  # 统计有效位置数量
        # 5. 避免除零错误并计算准确率
        if total_valid == 0:
            return 0.0  # 如果没有有效token，返回0准确率
        accuracy = correct_count / total_valid
        return accuracy

    def train_model(self, use_positional_encoding=True, model_name="baseline"):
        """
        训练单个模型：
        这个方法是消融实验的核心，负责训练一个完整的Transformer模型，
        并根据消融设置控制是否使用位置编码。
        Args:
            use_positional_encoding: 是否使用位置编码（消融变量）
                - True: 使用标准位置编码（基线模型）
                - False: 不使用位置编码（消融模型）
            model_name: 模型名称标识，用于日志记录和结果存储
                - 示例: "with_positional_encoding", "without_positional_encoding"
        Returns:
            dict: 包含完整训练结果和模型状态的字典
                - model: 训练完成的模型实例
                - train_losses: 每个epoch的平均训练损失列表
                - val_losses: 每个epoch的平均验证损失列表
                - train_perplexities: 每个epoch的平均训练困惑度列表
                - val_perplexities: 每个epoch的平均验证困惑度列表
                - train_accuracies: 每个epoch的平均训练准确率列表
                - val_accuracies: 每个epoch的平均验证准确率列表
                - rouge_scores: 每个epoch的ROUGE分数列表
                - best_accuracy: 最佳验证准确率
                - best_epoch: 最佳准确率出现的epoch
                - best_model_state: 最佳模型状态字典（用于保存和恢复）
        """
        print(f"\n{'=' * 60}")
        print(f"训练模型: {model_name}")
        print(f"使用位置编码: {use_positional_encoding}")
        print(f"{'=' * 60}")

        # 创建模型 - 使用消融实验专用Transformer
        model = AblationTransformer(
            self.config,
            self.tokenizer,
            use_positional_encoding=use_positional_encoding
        ).to(self.device)  # 将模型移动到指定设备（GPU/CPU）

        # 优化器 - AdamW (Adam + 权重衰减)
        # AdamW结合了Adam的自适应学习率和权重衰减正则化
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        # 学习率调度器 - 余弦退火
        # 按照余弦函数调整学习率，从初始值衰减到0
        # T_max参数指定余弦周期的长度（总训练轮数）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )

        # 损失函数 - 交叉熵损失
        # ignore_index参数忽略填充token，避免填充位置影响损失计算
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # 训练记录 - 初始化所有跟踪指标
        train_losses = []  # 训练损失历史（每个epoch的平均值）
        val_losses = []  # 验证损失历史
        train_perplexities = []  # 训练困惑度历史
        val_perplexities = []  # 验证困惑度历史
        train_accuracies = []  # 训练准确率历史
        val_accuracies = []  # 验证准确率历史
        rouge_scores = []  # ROUGE分数历史（每个epoch）

        # 基于准确率选择最佳模型
        best_accuracy = 0.0  # 记录最佳验证准确率
        best_model_state = None  # 记录最佳模型状态（用于保存）
        best_epoch = 0  # 记录最佳准确率出现的epoch

        # 训练循环 - 遍历所有epoch
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # ==================== 训练阶段 ====================
            model.train()  # 设置为训练模式（启用dropout、batch normalization等）
            total_train_loss = 0
            total_train_perplexity = 0
            total_train_accuracy = 0

            # 使用tqdm进度条显示训练进度
            progress_bar = tqdm(self.train_loader, desc=f'训练 {model_name}')

            # 遍历训练数据的所有批次
            for batch in progress_bar:
                # 准备数据 - 将数据移动到指定设备
                input_ids = batch['input_ids'].to(self.device)  # 输入序列token IDs
                labels = batch['labels'].to(self.device)  # 目标序列token IDs

                # 前向传播 - 使用Teacher Forcing技术
                # Teacher Forcing: 使用真实目标序列作为解码器输入，加速收敛
                # labels[:, :-1] 移除目标序列的最后一个token，作为解码器输入
                # 这样模型学习的是根据前n-1个token预测第n个token
                outputs = model(input_ids, labels[:, :-1])
                logits = outputs['logits']  # 模型输出的原始分数

                # 计算损失 - 交叉熵损失
                # 将logits重塑为二维：(batch_size * seq_len, vocab_size)
                # 将labels重塑为一维：(batch_size * seq_len)
                # labels[:, 1:] 移除第一个token（通常是起始符）
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1)
                )

                # 计算准确率 - 正确预测的token比例
                accuracy = self.calculate_accuracy(
                    logits,
                    labels[:, 1:],  # 忽略第一个token
                    ignore_index=self.tokenizer.pad_token_id
                )

                # 反向传播
                optimizer.zero_grad()  # 清空上一轮的梯度
                loss.backward()  # 反向传播计算梯度
                # 梯度裁剪 - 防止梯度爆炸
                # 当梯度范数超过阈值时，将梯度按比例缩小
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()  # 根据梯度更新模型参数

                # 累计统计 - 用于计算epoch平均值
                total_train_loss += loss.item()
                total_train_perplexity += self.calculate_perplexity(loss.item())
                total_train_accuracy += accuracy

                # 更新进度条显示 - 实时显示当前批次的指标
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',  # 当前批次损失
                    'acc': f'{accuracy:.4f}',  # 当前批次准确率
                    'ppl': f'{self.calculate_perplexity(loss.item()):.2f}'  # 当前批次困惑度
                })

            # 计算训练平均值 - 整个epoch的平均值
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_train_perplexity = total_train_perplexity / len(self.train_loader)
            avg_train_accuracy = total_train_accuracy / len(self.train_loader)

            # 记录训练结果
            train_losses.append(avg_train_loss)
            train_perplexities.append(avg_train_perplexity)
            train_accuracies.append(avg_train_accuracy)

            # ==================== 验证阶段 ====================
            # 在验证集上评估模型性能，不更新参数
            avg_val_loss, avg_val_perplexity, avg_val_accuracy, rouge = self.validate_model(
                model, self.val_loader, criterion
            )

            # 记录验证结果
            val_losses.append(avg_val_loss)
            val_perplexities.append(avg_val_perplexity)
            val_accuracies.append(avg_val_accuracy)
            rouge_scores.append(rouge)

            # 学习率调度 - 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']# 获取当前学习率

            # 打印结果
            print(f"训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
            print(f"训练困惑度: {avg_train_perplexity:.2f} | 验证困惑度: {avg_val_perplexity:.2f}")
            print(f"训练准确率: {avg_train_accuracy:.4f} | 验证准确率: {avg_val_accuracy:.4f}")
            print(f"学习率: {current_lr:.2e}")
            print(f"ROUGE-1: {rouge['rouge1']:.4f} | ROUGE-2: {rouge['rouge2']:.4f} | ROUGE-L: {rouge['rougeL']:.4f}")

            # 基于验证准确率保存最佳模型
            if avg_val_accuracy > best_accuracy:
                best_accuracy = avg_val_accuracy
                best_epoch = epoch + 1
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_perplexity': avg_train_perplexity,
                    'val_perplexity': avg_val_perplexity,
                    'train_accuracy': avg_train_accuracy,
                    'val_accuracy': avg_val_accuracy,
                    'rouge_scores': rouge
                }
                print(f"✅ 新的最佳模型 (Epoch {best_epoch}, 验证准确率: {best_accuracy:.4f})")

        # 返回完整的训练结果
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_perplexities': train_perplexities,
            'val_perplexities': val_perplexities,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'rouge_scores': rouge_scores,
            'best_accuracy': best_accuracy,
            'best_epoch': best_epoch,
            'best_model_state': best_model_state
        }

    def validate_model(self, model, val_loader, criterion):
        """
        验证模型性能：
        在验证集上评估模型表现，不更新模型参数，只计算评估指标。
        这个方法用于在训练过程中监控模型性能，防止过拟合。
        Args:
            model: 要评估的模型实例
                - 类型: AblationTransformer
                - 状态: 已训练或正在训练中的模型
            val_loader: 验证数据加载器
                - 包含验证数据集的所有批次
                - 批次格式: {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
            criterion: 损失函数
                - 类型: nn.CrossEntropyLoss
                - 配置: ignore_index=self.tokenizer.pad_token_id
        Returns:
            tuple: (avg_loss, avg_perplexity, avg_accuracy, rouge)
                - avg_loss: 平均验证损失（标量）
                - avg_perplexity: 平均验证困惑度（标量）
                - avg_accuracy: 平均验证准确率（标量）
                - rouge: ROUGE评估分数（字典）
                    - 包含rouge1, rouge2, rougeL三个分数
        """
        model.eval()
        # 初始化累计变量
        total_loss = 0  # 累计验证损失
        total_perplexity = 0  # 累计验证困惑度
        total_accuracy = 0  # 累计验证准确率
        # 用于ROUGE评估的预测和参考文本
        all_predictions = []  # 存储模型生成的所有摘要文本
        all_references = []  # 存储所有真实的摘要文本（参考摘要）

        with torch.no_grad():# 禁用梯度计算，节省内存并加速推理过程
            # 使用tqdm进度条遍历验证集的所有批次
            for batch in tqdm(val_loader, desc='验证'):
                # 准备数据 - 将数据移动到指定设备
                input_ids = batch['input_ids'].to(self.device)  # 输入序列token IDs
                labels = batch['labels'].to(self.device)  # 目标序列token IDs

                # 前向传播
                outputs = model(input_ids, labels[:, :-1])
                logits = outputs['logits']

                # 计算损失
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1)
                )

                # 计算准确率
                accuracy = self.calculate_accuracy(
                    logits,
                    labels[:, 1:],
                    ignore_index=self.tokenizer.pad_token_id
                )

                total_loss += loss.item()
                total_perplexity += self.calculate_perplexity(loss.item())
                total_accuracy += accuracy

                # 生成预测 - 用于ROUGE评估
                # 使用模型生成完整的摘要，而不是Teacher Forcing
                try:
                    # 创建源序列掩码 - 标识哪些位置是真实内容
                    src_mask = model._create_src_mask(input_ids)
                    # 使用模型生成摘要（推理模式）
                    # generate方法使用自回归方式生成完整序列
                    predictions = model.generate(
                        input_ids,  # 输入序列
                        src_mask=src_mask,  # 源序列掩码
                        max_length=self.config.max_target_length  # 最大生成长度
                    )

                    # 解码预测结果 - 将token ID转换为文本
                    decoded_preds = self.tokenizer.batch_decode(
                        predictions.cpu(),  # 将张量移动到CPU
                        skip_special_tokens=True  # 跳过特殊token（如[CLS], [SEP], [PAD]）
                    )

                    # 解码真实标签
                    decoded_labels = self.tokenizer.batch_decode(
                        labels.cpu(),  # 将张量移动到CPU
                        skip_special_tokens=True  # 跳过特殊token
                    )

                    # 存储结果用于后续评估
                    all_predictions.extend(decoded_preds)  # 添加生成的摘要
                    all_references.extend(decoded_labels)  # 添加真实的摘要
                except Exception as e:
                    print(f"生成预测时出错: {e}")
                    continue

        # 计算平均指标 - 整个验证集的平均值
        avg_loss = total_loss / len(val_loader)  # 平均损失
        avg_perplexity = total_perplexity / len(val_loader)  # 平均困惑度
        avg_accuracy = total_accuracy / len(val_loader)  # 平均准确率

        # 计算ROUGE分数 - 评估生成摘要的质量
        if len(all_predictions) > 0 and len(all_references) > 0:
            # 成功生成预测，计算ROUGE分数
            rouge = self.calculate_rouge(all_predictions, all_references)
        else:
            # 无法生成有效预测，使用默认分数
            rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            print("警告：无法计算ROUGE分数，预测结果为空")

        return avg_loss, avg_perplexity, avg_accuracy, rouge

    def calculate_rouge(self, predictions, references):
        """
        计算ROUGE评估分数 - 用于评估生成摘要的质量
        Args:
            predictions (list[str]): 模型生成的摘要文本列表
                - 每个元素是一个字符串，代表模型对单个样本的摘要生成结果
                - 示例: ["猫在垫子上睡觉", "狗在院子里玩耍"]
                - 要求: 必须与references列表顺序一致且长度相同
            references (list[str]): 人工撰写的参考摘要文本列表
                - 每个元素是一个字符串，代表该样本的标准答案（黄金摘要）
                - 示例: ["一只猫在垫子上休息", "狗在花园里玩"]
                - 要求: 必须与predictions列表顺序一致且长度相同
        Returns:
            dict: 包含三种ROUGE分数的字典
                - 'rouge1': ROUGE-1分数（基于单个单词）
                - 'rouge2': ROUGE-2分数（基于单词对）
                - 'rougeL': ROUGE-L分数（基于最长公共子序列）
                - 每个分数都是该批次所有样本的平均值，范围[0.0, 1.0]
        Note:
            ROUGE分数越高表示生成摘要与参考摘要越相似，质量越好
            通常ROUGE-1 > ROUGE-L > ROUGE-2，因为匹配要求逐渐严格
        """
        # 初始化ROUGE评分器
        # RougeScorer是rouge_score库的核心类，负责计算各种ROUGE指标
        # 参数说明：
        # - ['rouge1', 'rouge2', 'rougeL']: 同时计算三种ROUGE分数
        #   * rouge1: 基于unigram（单个单词）的重叠度
        #   * rouge2: 基于bigram（连续两个单词）的重叠度
        #   * rougeL: 基于最长公共子序列（LCS）的相似度
        # - use_stemmer=True: 对单词进行词干提取（如"running"→"run"）
        #   减少词形变化对匹配的影响，提高评估的语义准确性
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],  # 指定要计算的ROUGE类型
            use_stemmer=True  # 启用词干提取，提高匹配准确性
        )

        # 初始化分数存储列表
        rouge1_scores = []  # 存储每个样本的ROUGE-1分数
        rouge2_scores = []  # 存储每个样本的ROUGE-2分数
        rougeL_scores = []  # 存储每个样本的ROUGE-L分数

        # 遍历每个样本对（生成摘要 vs 参考摘要）
        for pred, ref in zip(predictions, references):
            # 计算当前样本的ROUGE分数
            # scorer.score() 返回包含precision, recall, fmeasure的字典
            scores = scorer.score(ref, pred)
            # scorer.score()返回的三种指标：
            # scores = {
            #     'rouge1': {
            #         'precision': 0.75,   # 精确率：生成摘要中有多少是正确的
            #         'recall': 0.60,      # 召回率：参考摘要中有多少被覆盖
            #         'fmeasure': 0.67     # F1分数：精确率和召回率的调和平均
            #     },
            #     'rouge2': {
            #         'precision': 0.50,
            #         'recall': 0.40,
            #         'fmeasure': 0.44
            #     },
            #     'rougeL': {
            #         'precision': 0.70,
            #         'recall': 0.55,
            #         'fmeasure': 0.62
            #     }
            # }

            # 提取并存储F1分数（精确率和召回率的调和平均）
            # F1分数综合考虑了生成摘要的准确性和完整性
            rouge1_scores.append(scores['rouge1'].fmeasure)  # 单词级别相似度
            rouge2_scores.append(scores['rouge2'].fmeasure)  # 短语级别相似度
            rougeL_scores.append(scores['rougeL'].fmeasure)  # 句子结构相似度

        # 计算批次平均分数并返回
        return {
            'rouge1': np.mean(rouge1_scores),  # 平均ROUGE-1分数
            'rouge2': np.mean(rouge2_scores),  # 平均ROUGE-2分数
            'rougeL': np.mean(rougeL_scores)  # 平均ROUGE-L分数
        }

    def run_study(self):
        """
        运行完整的消融实验
        这个方法执行整个位置编码消融实验，包括：
        1. 训练基线模型（使用位置编码）
        2. 训练消融模型（不使用位置编码）
        3. 保存所有实验结果
        4. 生成可视化比较图表
        Returns:
            dict: 完整的实验结果字典
                - 包含两个实验的所有评估指标和历史记录
                - 结构: {
                    "with_positional_encoding": {...},    # 基线模型结果
                    "without_positional_encoding": {...}  # 消融模型结果
                }
        """
        print("开始位置编码消融实验")
        print("=" * 60)

        # 实验1: 使用位置编码（基线模型）
        print("\n🔬 实验1: 训练带有位置编码的模型（基线）")
        # 训练基线模型
        baseline_results = self.train_model(
            use_positional_encoding=True,  # 使用位置编码
            model_name="with_positional_encoding"  # 模型名称标识
        )
        # 存储基线模型结果 - 完整记录所有评估指标
        self.results["with_positional_encoding"] = {
            "final_train_loss": baseline_results['train_losses'][-1],
            "final_val_loss": baseline_results['val_losses'][-1],
            "final_train_perplexity": baseline_results['train_perplexities'][-1],
            "final_val_perplexity": baseline_results['val_perplexities'][-1],
            "final_train_accuracy": baseline_results['train_accuracies'][-1],
            "final_val_accuracy": baseline_results['val_accuracies'][-1],
            "final_rouge": baseline_results['rouge_scores'][-1],
            "best_accuracy": baseline_results['best_accuracy'],
            "best_epoch": baseline_results['best_epoch'],
            "all_rouge_scores": baseline_results['rouge_scores'],
            "train_losses": baseline_results['train_losses'],
            "val_losses": baseline_results['val_losses'],
            "train_perplexities": baseline_results['train_perplexities'],
            "val_perplexities": baseline_results['val_perplexities'],
            "train_accuracies": baseline_results['train_accuracies'],
            "val_accuracies": baseline_results['val_accuracies']
        }

        # 实验2: 不使用位置编码（消融模型）
        print("\n🔬 实验2: 训练不带有位置编码的模型（消融）")
        # 训练消融模型
        ablation_results = self.train_model(
            use_positional_encoding=False,  # 不使用位置编码
            model_name="without_positional_encoding"  # 模型名称标识
        )
        # 存储消融模型结果 - 结构同上
        self.results["without_positional_encoding"] = {
            "final_train_loss": ablation_results['train_losses'][-1],
            "final_val_loss": ablation_results['val_losses'][-1],
            "final_train_perplexity": ablation_results['train_perplexities'][-1],
            "final_val_perplexity": ablation_results['val_perplexities'][-1],
            "final_train_accuracy": ablation_results['train_accuracies'][-1],
            "final_val_accuracy": ablation_results['val_accuracies'][-1],
            "final_rouge": ablation_results['rouge_scores'][-1],
            "best_accuracy": ablation_results['best_accuracy'],
            "best_epoch": ablation_results['best_epoch'],
            "all_rouge_scores": ablation_results['rouge_scores'],
            "train_losses": ablation_results['train_losses'],
            "val_losses": ablation_results['val_losses'],
            "train_perplexities": ablation_results['train_perplexities'],
            "val_perplexities": ablation_results['val_perplexities'],
            "train_accuracies": ablation_results['train_accuracies'],
            "val_accuracies": ablation_results['val_accuracies']
        }

        # 保存结果
        self.save_results()

        # 可视化比较
        self.visualize_comparison()

        return self.results

    def save_results(self):
        """
        保存实验结果到文件
        将完整的实验结果保存为两种格式：
        1. JSON格式：完整的数据结构，便于程序读取和分析
        2. 文本格式：人类可读的摘要报告，便于快速查看结果
        文件命名约定：
            - ablation_results_{epochs}epochs_{heads}heads.json
            - experiment_summary_{epochs}epochs_{heads}heads.txt
        其中{epochs}和{heads}是配置参数，便于区分不同实验设置的结果
        """
        # 保存JSON结果
        results_file = os.path.join(self.results_dir,f"ablation_results_{self.config.num_epochs}epochs_{self.config.n_heads}heads.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # indent=2: 美化格式，便于阅读
            # ensure_ascii=False: 支持中文字符
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # 保存文本摘要
        summary_file = os.path.join(self.results_dir,f"experiment_summary_{self.config.num_epochs}epochs_{self.config.n_heads}heads.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("位置编码消融实验摘要\n")
            f.write("=" * 50 + "\n\n")

            baseline = self.results["with_positional_encoding"]
            ablation = self.results["without_positional_encoding"]

            f.write("基线模型（有位置编码）:\n")
            f.write(f"  最终训练损失: {baseline['final_train_loss']:.4f}\n")
            f.write(f"  最终验证损失: {baseline['final_val_loss']:.4f}\n")
            f.write(f"  最终训练困惑度: {baseline['final_train_perplexity']:.2f}\n")
            f.write(f"  最终验证困惑度: {baseline['final_val_perplexity']:.2f}\n")
            f.write(f"  最终训练准确率: {baseline['final_train_accuracy']:.4f}\n")
            f.write(f"  最终验证准确率: {baseline['final_val_accuracy']:.4f}\n")
            f.write(f"  最终ROUGE-1: {baseline['final_rouge']['rouge1']:.4f}\n")
            f.write(f"  最佳验证准确率: {baseline['best_accuracy']:.4f} (Epoch {baseline['best_epoch']})\n\n")

            f.write("消融模型（无位置编码）:\n")
            f.write(f"  最终训练损失: {ablation['final_train_loss']:.4f}\n")
            f.write(f"  最终验证损失: {ablation['final_val_loss']:.4f}\n")
            f.write(f"  最终训练困惑度: {ablation['final_train_perplexity']:.2f}\n")
            f.write(f"  最终验证困惑度: {ablation['final_val_perplexity']:.2f}\n")
            f.write(f"  最终训练准确率: {ablation['final_train_accuracy']:.4f}\n")
            f.write(f"  最终验证准确率: {ablation['final_val_accuracy']:.4f}\n")
            f.write(f"  最终ROUGE-1: {ablation['final_rouge']['rouge1']:.4f}\n")
            f.write(f"  最佳验证准确率: {ablation['best_accuracy']:.4f} (Epoch {ablation['best_epoch']})\n\n")

            # 计算差异
            accuracy_diff = baseline['final_val_accuracy'] - ablation['final_val_accuracy']
            rouge1_diff = baseline['final_rouge']['rouge1'] - ablation['final_rouge']['rouge1']
            rouge2_diff = baseline['final_rouge']['rouge2'] - ablation['final_rouge']['rouge2']
            rougeL_diff = baseline['final_rouge']['rougeL'] - ablation['final_rouge']['rougeL']

            # 差异分析
            f.write("性能差异（基线 - 消融）:\n")
            f.write(f"  验证准确率差异: {accuracy_diff:+.4f} ({accuracy_diff / baseline['final_val_accuracy']:+.1%})\n")
            f.write(f"  ROUGE-1差异: {rouge1_diff:+.4f} ({rouge1_diff / baseline['final_rouge']['rouge1']:+.1%})\n")
            f.write(f"  ROUGE-2差异: {rouge2_diff:+.4f} ({rouge2_diff / baseline['final_rouge']['rouge2']:+.1%})\n")
            f.write(f"  ROUGE-L差异: {rougeL_diff:+.4f} ({rougeL_diff / baseline['final_rouge']['rougeL']:+.1%})\n")

        print(f"✅ 实验结果已保存到: {results_file}")
        print(f"✅ 实验摘要已保存到: {summary_file}")

    def visualize_comparison(self):
        """
        可视化比较结果 - 生成4个子图的综合比较图表
        图表包含：
            1. 训练损失比较曲线
            2. 验证准确率比较曲线
            3. 验证困惑度比较曲线
            4. 最终性能柱状图比较
        使用2x2网格布局，便于全面比较两个模型的性能差异
        """
        # 设置绘图风格
        # plt.style.use('seaborn-v0_8')  # 使用seaborn风格，图表更美观
        # 创建2x2的子图网格
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.4)  # 增加子图之间的垂直间距（默认约0.2）

        # 提取两个模型的结果数据
        baseline = self.results["with_positional_encoding"]
        ablation = self.results["without_positional_encoding"]

        # 创建epoch序列（从1开始）
        epochs = range(1, len(baseline['train_losses']) + 1)

        # ==================== 1. 训练损失比较曲线（左上） ====================
        axes[0, 0].plot(epochs, baseline['train_losses'], 'b-', label='有位置编码', linewidth=2)
        axes[0, 0].plot(epochs, ablation['train_losses'], 'r-', label='无位置编码', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('训练损失')
        axes[0, 0].set_title('训练损失比较', pad=10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)  # 添加网格线，alpha控制透明度

        # ==================== 2. 验证准确率比较曲线（右上） ====================
        axes[0, 1].plot(epochs, baseline['val_accuracies'], 'b-', label='有位置编码', linewidth=2)
        axes[0, 1].plot(epochs, ablation['val_accuracies'], 'r-', label='无位置编码', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('验证准确率')
        axes[0, 1].set_title('验证准确率比较', pad=10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # ==================== 3. 验证困惑度比较曲线（左下） ====================
        axes[1, 0].plot(epochs, baseline['val_perplexities'], 'b-', label='有位置编码', linewidth=2)
        axes[1, 0].plot(epochs, ablation['val_perplexities'], 'r-', label='无位置编码', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('验证困惑度')
        axes[1, 0].set_title('验证困惑度比较', pad=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # ==================== 4. 最终性能柱状图比较（右下） ====================
        metrics = ['准确率', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        baseline_scores = [
            baseline['final_val_accuracy'],
            baseline['final_rouge']['rouge1'],
            baseline['final_rouge']['rouge2'],
            baseline['final_rouge']['rougeL']
        ]
        ablation_scores = [
            ablation['final_val_accuracy'],
            ablation['final_rouge']['rouge1'],
            ablation['final_rouge']['rouge2'],
            ablation['final_rouge']['rougeL']
        ]

        # 设置柱状图位置和宽度
        x = np.arange(len(metrics))  # x轴位置：[0, 1, 2, 3]
        width = 0.35  # 柱状图宽度
        # 绘制柱状图
        axes[1, 1].bar(x - width / 2, baseline_scores, width, label='有位置编码', alpha=0.8)
        axes[1, 1].bar(x + width / 2, ablation_scores, width, label='无位置编码', alpha=0.8)
        axes[1, 1].set_xlabel('评估指标')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].set_title('最终性能比较', pad=10)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 调整子图间距，避免重叠
        plt.tight_layout()
        # 保存图表为高分辨率PNG文件
        plot_file = os.path.join(self.results_dir, f"ablation_comparison_{self.config.num_epochs}epochs_{self.config.n_heads}heads.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        # dpi=300: 高分辨率（300点每英寸）
        # bbox_inches='tight': 自动调整边界，避免裁剪
        plt.show()

        print(f"✅ 比较图表已保存到: {plot_file}")


def main():
    """主函数：运行消融实验"""
    # 设置随机种子
    set_seed(config.seed)
    print("设置随机种子完成")

    # 获取数据
    print("加载数据...")
    train_loader, val_loader, test_loader, tokenizer = get_data_loaders(config)

    # 运行消融实验
    study = AblationStudy(config, tokenizer, train_loader, val_loader, test_loader)
    results = study.run_study()

    print("\n🎉 消融实验完成！")
    print(f"结果保存在: {study.results_dir}")


if __name__ == "__main__":
    main()