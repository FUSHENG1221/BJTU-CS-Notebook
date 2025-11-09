import math
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

from model import Transformer
from data_loader import get_data_loaders
from config import config
from utils import set_seed

import warnings
warnings.filterwarnings("ignore")# 忽略警告输出
plt.rcParams["font.sans-serif"] = ["SimHei"]# 设置matplotlib中文字体显示（解决中文乱码问题）
plt.rcParams["axes.unicode_minus"] = False# 显示负号

class Trainer:
    def __init__(self, model, train_loader, val_loader,test_loader, tokenizer, config):
        """
        训练器初始化
        Args:
            model: Transformer模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            tokenizer: 分词器
            config: 配置对象
        """
        self.model = model.to(config.device)# 将模型移动到指定设备（GPU/CPU），确保模型参数和计算在正确的硬件上执行
        self.train_loader = train_loader# 存储训练数据加载器，用于迭代训练数据批次
        self.val_loader = val_loader# 存储验证数据加载器，用于在训练过程中评估模型性能
        self.test_loader = test_loader# 存储测试数据加载器，用于在训练过程中评估模型性能
        self.tokenizer = tokenizer# 存储分词器实例，用于文本编码和解码操作
        self.config = config# 存储配置对象，包含所有训练超参数和设置

        # 优化器和损失函数
        # 初始化AdamW优化器，结合了Adam算法的自适应学习率和权重衰减正则化
        # model.parameters()返回模型中所有需要训练的参数
        # lr设置初始学习率，根据config.learning_rate确定
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        # 初始化余弦退火学习率调度器，按照余弦函数调整学习率
        # T_max参数指定余弦周期的长度（这里设置为总训练轮数）
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        # 初始化交叉熵损失函数，用于计算模型预测与真实标签之间的差异
        # ignore_index参数指定要忽略的标签索引（这里忽略填充token的ID）
        # 这样可以避免填充位置对损失计算产生影响
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # 训练记录
        self.train_losses = []# 初始化训练损失记录列表，用于存储每个epoch的平均训练损失
        self.val_losses = []# 初始化验证损失记录列表，用于存储每个epoch的验证损失
        self.rouge_scores = []# 初始化ROUGE分数记录列表，用于存储每个epoch的文本生成质量评估分数
        # ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是专门用于评估自动文摘和机器翻译等文本生成任务的质量指标。
        # ROUGE-1（基于单词）：比较单个单词的重叠情况
        # ROUGE-2（基于单词对）:比较连续两个单词的组合
        # ROUGE-L（基于最长公共子序列）:# 寻找最长的连续匹配序列
        self.train_perplexities = []  # 新增：训练困惑度
        self.val_perplexities = []  # 新增：验证困惑度
        self.train_accuracies = []  # 新增：训练准确率
        self.val_accuracies = []  # 新增：验证准确率

    def calculate_perplexity(self, loss):
        """
        计算困惑度（Perplexity）
        困惑度是语言模型中最重要的评估指标之一，表示模型对测试数据的"困惑程度"。
        数学公式：perplexity = exp(loss)
        Args:
            loss: 交叉熵损失值
        Returns:
            float: 困惑度值
        解释：
        - 困惑度越低，说明模型对数据的预测越准确
        - 完美预测：困惑度 = 1（损失为0时）
        - 随机猜测：困惑度 = 词汇表大小（最差情况）
        """
        return math.exp(loss)

    def calculate_accuracy(self, logits, labels, ignore_index=-100):
        """
        计算准确率（Accuracy）:
        准确率衡量模型预测正确的token比例，是直观的评估指标。
        Args:
            logits: 模型输出，形状为(batch_size, seq_len, vocab_size)
            labels: 真实标签，形状为(batch_size, seq_len)
            ignore_index: 要忽略的标签索引（如填充token）
        Returns:
            float: 准确率（0.0到1.0之间）
        """
        # 获取预测结果（概率最大的token）
        predictions = torch.argmax(logits, dim=-1)
        # 创建有效token的掩码（忽略填充token）
        valid_mask = (labels != ignore_index)
        # 计算正确预测的数量
        correct = (predictions == labels) & valid_mask
        correct_count = correct.sum().item()
        # 计算有效token的总数
        total_valid = valid_mask.sum().item()
        # 避免除零错误
        if total_valid == 0:
            return 0.0
        accuracy = correct_count / total_valid
        return accuracy

    def train_epoch(self, epoch):
        """
        训练一个完整的epoch（遍历整个训练数据集一次）
        Args:
            epoch (int): 当前训练轮数，用于进度显示和日志记录
        Returns:
            float（avg_loss）: 该epoch的平均训练损失，用于监控训练效果
        Note:
            一个epoch包含对训练数据集中所有批次的前向传播、损失计算、反向传播和参数更新
        """
        # 将模型设置为训练模式
        # 这会启用dropout、batch normalization等训练特有的行为
        self.model.train()
        total_loss = 0# 初始化该epoch的总损失，用于后续计算平均损失
        total_perplexity = 0
        total_accuracy = 0
        total_batches = 0
        # 创建进度条，显示当前epoch和训练进度；tqdm提供了直观的训练进度可视化
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        # 遍历训练数据加载器中的每个批次
        for batch in progress_bar:
            # 1. 数据准备阶段 - 将数据移动到指定设备（GPU/CPU）
            # 将输入序列的token ID移动到模型所在的设备
            input_ids = batch['input_ids'].to(self.config.device)
            # 注意力掩码，标识哪些位置是真实内容（1）和填充内容（0）
            attention_mask = batch['attention_mask'].to(self.config.device)
            # 目标序列（摘要文本）的token ID，作为训练标签
            labels = batch['labels'].to(self.config.device)

            # 2. 前向传播（使用Teacher Forcing技术）
            # Teacher Forcing：使用真实目标序列作为解码器输入，加速收敛
            # labels[:, :-1] 移除目标序列的最后一个token，作为解码器输入
            # 这样模型学习的是根据前n-1个token预测第n个token
            outputs = self.model(input_ids, labels[:, :-1])
            # 模型输出的原始分数（logits），形状为(batch_size, seq_len, vocab_size)
            logits = outputs['logits']

            # 3. 损失计算 - 交叉熵损失：比较模型预测与真实标签的差异
            # logits.reshape(-1, logits.size(-1)),
            # 将logits重塑为二维张量：(batch_size * seq_len, vocab_size)
            # 这样每一行对应一个位置的词汇表概率分布
            # labels[:, 1:].reshape(-1):
            # 将目标序列重塑为一维张量：(batch_size * seq_len)
            # labels[:, 1:] 移除目标序列的第一个token（通常是起始符）
            # 因为模型应该从第二个token开始预测（第一个token是已知的起始点）
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )

            # 计算准确率
            accuracy = self.calculate_accuracy(
                logits,
                labels[:, 1:],
                ignore_index=self.tokenizer.pad_token_id
            )

            # 4. 反向传播和参数更新
            # 清空上一轮计算的梯度，防止梯度累积
            self.optimizer.zero_grad()
            # 反向传播计算梯度：损失函数对模型参数的导数
            loss.backward()

            # 梯度裁剪 - 防止梯度爆炸
            # 当梯度范数超过阈值时，将梯度按比例缩小。这有助于训练稳定性，特别是处理长序列时
            # 过大的梯度会导致参数更新过大，训练不稳定甚至发散
            # 参数1：要裁剪的模型参数；参数2：最大梯度范数阈值
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            # 根据梯度更新模型参数
            self.optimizer.step()

            # 5. 损失记录和进度更新
            # 累加该批次的损失值（.item()将张量转换为Python数值）
            total_loss += loss.item()
            total_perplexity += self.calculate_perplexity(loss.item())
            total_accuracy += accuracy
            total_batches += 1
            # 更新进度条显示当前批次的损失
            # progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'ppl': f'{self.calculate_perplexity(loss.item()):.2f}'
            })

        # 计算该epoch的平均损失（总损失除以批次数量）
        avg_loss = total_loss / len(self.train_loader)
        avg_perplexity = total_perplexity / len(self.train_loader)
        avg_accuracy = total_accuracy / len(self.train_loader)
        # 将平均损失记录到训练历史中，用于后续分析和可视化
        self.train_losses.append(avg_loss)
        self.train_perplexities.append(avg_perplexity)
        self.train_accuracies.append(avg_accuracy)
        # 返回平均损失，供外部监控训练进度
        # return avg_loss
        return avg_loss, avg_perplexity, avg_accuracy

    def validate(self, epoch):
        """
        验证模型性能 - 在验证集上评估模型表现，并保存预测结果
        Args:
            epoch (int): 当前训练轮数，用于进度显示和日志记录
        Returns:
            tuple: (avg_loss, rouge_scores)
                - avg_loss (float): 平均验证损失
                - rouge_scores (dict): 包含ROUGE-1、ROUGE-2、ROUGE-L分数的字典
        Note:
            验证过程不更新模型参数，仅用于评估和监控训练进度
            使用torch.no_grad()禁用梯度计算，节省内存和计算资源
        """
        # 将模型设置为评估模式
        # 这会禁用dropout、batch normalization等训练特有的行为，确保评估结果的一致性
        self.model.eval()
        # 初始化累计变量
        total_loss = 0  # 累计验证损失
        total_perplexity = 0
        total_accuracy = 0
        all_predictions = []  # 存储所有生成的摘要文本
        all_references = []  # 存储所有真实的摘要文本

        # 使用torch.no_grad()上下文管理器；禁用梯度计算，节省内存并加速推理过程
        with torch.no_grad():
            # 使用tqdm进度条遍历验证集的所有批次
            for batch in tqdm(self.val_loader, desc='Validating'):
                # 1. 数据准备 - 将数据移动到指定设备
                input_ids = batch['input_ids'].to(self.config.device)  # 输入序列token IDs
                attention_mask = batch['attention_mask'].to(self.config.device)  # 注意力掩码
                labels = batch['labels'].to(self.config.device)  # 目标序列token IDs

                # 2. 前向传播计算损失（使用Teacher Forcing）
                # 输入：对话文本，目标：摘要文本（去掉最后一个token）
                outputs = self.model(input_ids, labels[:, :-1])
                logits = outputs['logits']  # 模型输出的原始分数

                # 3. 计算验证损失
                loss = self.criterion(
                    # 将logits重塑为二维：(batch_size * seq_len, vocab_size)
                    logits.reshape(-1, logits.size(-1)),
                    # 将目标序列重塑为一维：(batch_size * seq_len)
                    # labels[:, 1:] 移除第一个token（通常是起始符）
                    labels[:, 1:].reshape(-1)
                )

                # 计算准确率
                accuracy = self.calculate_accuracy(
                    logits,
                    labels[:, 1:],
                    ignore_index=self.tokenizer.pad_token_id
                )

                total_loss += loss.item()  # 累加批次损失
                total_perplexity += self.calculate_perplexity(loss.item())
                total_accuracy += accuracy

                # 4. 生成预测（用于ROUGE评估）
                try:
                    # 4.1 创建源序列掩码（忽略填充token）
                    src_mask = self.model._create_src_mask(input_ids)

                    # 4.2 使用模型生成摘要（推理模式）
                    predictions = self.model.generate(
                        input_ids,
                        src_mask=src_mask,
                        max_length=self.config.max_target_length
                    )

                    # 4.3 解码预测结果和真实标签
                    decoded_preds = self.tokenizer.batch_decode(
                        predictions.cpu(),  # 将张量移动到CPU
                        skip_special_tokens=True  # 跳过特殊token（如[CLS], [SEP]）
                    )
                    decoded_labels = self.tokenizer.batch_decode(
                        labels.cpu(),  # 将张量移动到CPU
                        skip_special_tokens=True  # 跳过特殊token
                    )

                    # 4.4 存储结果用于后续评估
                    all_predictions.extend(decoded_preds)  # 添加生成的摘要
                    all_references.extend(decoded_labels)  # 添加真实的摘要

                except Exception as e:
                    print(f"生成预测时出错: {e}")
                    # 备用方案：使用贪心解码
                    try:
                        # 直接使用模型输出进行贪心解码
                        pred_tokens = torch.argmax(logits, dim=-1)
                        decoded_preds = self.tokenizer.batch_decode(
                            pred_tokens.cpu(),
                            skip_special_tokens=True
                        )
                        # 解码真实标签
                        decoded_labels = self.tokenizer.batch_decode(
                            labels.cpu(),
                            skip_special_tokens=True
                        )
                        all_predictions.extend(decoded_preds)
                        all_references.extend(decoded_labels)
                    except Exception as e2:
                        print(f"备用方案也失败: {e2}")
                        continue# 如果备用方案也失败，跳过该批次

        # 5. 计算ROUGE评估分数
        if len(all_predictions) > 0 and len(all_references) > 0:
            # 成功生成预测，计算ROUGE分数
            rouge = self.calculate_rouge(all_predictions, all_references)
        else:
            # 无法生成有效预测，使用默认分数
            rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            print("警告：无法计算ROUGE分数，预测结果为空")

        # 6. 计算平均验证损失
        avg_loss = total_loss / len(self.val_loader)  # 总损失除以批次数量
        avg_perplexity = total_perplexity / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)

        # 7. 记录验证结果
        self.val_losses.append(avg_loss)  # 记录验证损失历史
        self.val_perplexities.append(avg_perplexity)
        self.val_accuracies.append(avg_accuracy)
        self.rouge_scores.append(rouge)  # 记录ROUGE分数历史

        # 返回验证结果
        # return avg_loss, rouge
        return avg_loss, avg_perplexity, avg_accuracy, rouge

    def calculate_rouge(self, predictions, references):
        """
        计算ROUGE评估分数 - 用于评估生成摘要的质量
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 是
        自动文摘和机器翻译领域最常用的评估指标，通过比较生成文本与
        参考文本的n-gram重叠度来评估质量。
        Args:
            predictions (list[str]): 模型生成的摘要文本列表
                - 每个元素是一个字符串，代表模型对单个样本的摘要生成结果
                - 示例: ["猫在垫子上睡觉", "狗在院子里玩耍"]
            references (list[str]): 人工撰写的参考摘要文本列表
                - 每个元素是一个字符串，代表该样本的标准答案（黄金摘要）
                - 示例: ["一只猫在垫子上休息", "狗在花园里玩"]
                - 注意：必须与predictions列表顺序一致且长度相同
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
            # # scorer.score()返回的三种指标：
            # scores = {
            #     'rouge1': {
            #         'precision': 0.75,   # 精确率：生成摘要中有多少是正确的
            #         'recall': 0.60,      # 召回率：参考摘要中有多少被覆盖
            #         'fmeasure': 0.67     # F1分数：精确率和召回率的调和平均
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

    def train(self):
        """
        主训练循环 - 控制整个模型的训练过程
        这个方法是训练器的核心，负责：
        1. 初始化训练环境
        2. 执行多个训练epoch
        3. 在每个epoch后进行验证评估
        4. 保存最佳模型和检查点
        5. 输出训练结果和统计信息
        """
        print("开始训练...")
        print(f"设备: {self.config.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")

        best_rouge = 0  # 记录最佳的ROUGE-1分数
        best_accuracy = 0.0  # 记录最佳验证准确率
        best_epoch = 0  # 记录最佳性能出现的epoch

        for epoch in range(self.config.num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 50}")

            # 训练阶段
            # 执行一个完整的训练epoch
            # 使用epoch+1是为了显示更直观的进度（从1开始而不是0）
            # train_loss = self.train_epoch(epoch+1)
            train_loss, train_ppl, train_acc = self.train_epoch(epoch + 1)

            # 验证阶段
            # val_loss, rouge_scores = self.validate(epoch+1)
            val_loss, val_ppl, val_acc, rouge_scores = self.validate(epoch + 1)

            # 根据学习率调度策略调整学习率
            self.scheduler.step()  # 更新学习率（如余弦退火）
            current_lr = self.optimizer.param_groups[0]['lr']  # 获取当前学习率

            # 打印结果
            print(f"\n📊 训练结果:")
            print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            print(f"训练困惑度: {train_ppl:.2f} | 验证困惑度: {val_ppl:.2f}")
            print(f"训练准确率: {train_acc:.4f} | 验证准确率: {val_acc:.4f}")
            print(f"学习率: {current_lr:.2e}")
            print(f"ROUGE-1: {rouge_scores['rouge1']:.4f} | ROUGE-2: {rouge_scores['rouge2']:.4f} | ROUGE-L: {rouge_scores['rougeL']:.4f}")
            # print(f"\n训练结果:")
            # print(f"训练损失: {train_loss:.4f}")  # 训练集损失（越低越好）
            # print(f"验证损失: {val_loss:.4f}")  # 验证集损失（越低越好）
            # print(f"学习率: {current_lr:.2e}")  # 当前学习率（科学计数法显示）
            # # ROUGE分数输出（越高越好）
            # print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")  # 单词级别相似度
            # print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")  # 短语级别相似度
            # print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")  # 句子结构相似度

            # 保存最佳模型(基于准确率)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_epoch = epoch + 1  # 记录最佳epoch
                best_model_path = os.path.join(self.config.save_dir, f'best_model_{self.config.num_epochs}epochs_{self.config.n_heads}heads.pth')
                # 保存完整的模型检查点
                torch.save({
                    'epoch': epoch,  # 当前epoch数
                    'model_state_dict': self.model.state_dict(),  # 模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态
                    'train_loss': train_loss,  # 训练损失
                    'val_loss': val_loss,  # 验证损失
                    'train_perplexity': train_ppl,
                    'val_perplexity': val_ppl,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'rouge_scores': rouge_scores  # ROUGE分数
                }, best_model_path)  # 保存为best_model.pth文件
                print(f"✅ 保存新的最佳模型 (Epoch {best_epoch}, 验证准确率: {best_accuracy:.4f}, 保存地址：{best_model_path})")
            # 保存最佳模型（基于ROUGE-1分数）
            '''
            if rouge_scores['rouge1'] > best_rouge:
                best_rouge = rouge_scores['rouge1']  # 更新最佳分数
                best_epoch = epoch + 1  # 记录最佳epoch
                best_model_path = os.path.join(self.config.save_dir, 'best_model.pth')
                # 保存完整的模型检查点
                torch.save({
                    'epoch': epoch,  # 当前epoch数
                    'model_state_dict': self.model.state_dict(),  # 模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器状态
                    'train_loss': train_loss,  # 训练损失
                    'val_loss': val_loss,  # 验证损失
                    'train_perplexity': train_ppl,
                    'val_perplexity': val_ppl,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'rouge_scores': rouge_scores  # ROUGE分数
                }, best_model_path)  # 保存为best_model.pth文件
                print(f"✅ 保存新的最佳模型 (Epoch {best_epoch}, ROUGE-1: {best_rouge:.4f}, 保存地址：{best_model_path})")
            '''

            # 每5个epoch保存一次检查点（防止训练中断丢失进度）
            if (epoch + 1) % 5 == 0:
                # 检查点文件名
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'checkpoint_epoch_{epoch + 1}.pth'
                )
                # checkpoint_path = f'../results/model/checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'rouge_scores': rouge_scores
                }, checkpoint_path)
                print(f"💾 新的每五轮保存检查点: {checkpoint_path}")

        # 训练结束
        print(f"\n{'=' * 50}")
        print("训练完成!")
        print(f"{'=' * 50}")
        print(f"最佳模型在 Epoch {best_epoch}, ROUGE-1: {best_rouge:.4f}")

        # 绘制训练曲线
        self.plot_training_curves()

        # 保存最终模型
        torch.save(self.model.state_dict(), f'../results/model/final_model_{self.config.num_epochs}epochs_{self.config.n_heads}heads.pth')
        print(f"💾 保存最终模型: ../results/model/final_model_{self.config.num_epochs}epochs_{self.config.n_heads}heads.pth")

    def plot_training_curves(self):
        """绘制完整的训练曲线，包含所有评估指标"""
        plt.figure(figsize=(20, 10))

        # 1. 损失曲线
        plt.subplot(2, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 困惑度曲线
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_perplexities, 'b-', label='训练困惑度', linewidth=2)
        plt.plot(epochs, self.val_perplexities, 'r-', label='验证困惑度', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('困惑度曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. 准确率曲线
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_accuracies, 'b-', label='训练准确率', linewidth=2)
        plt.plot(epochs, self.val_accuracies, 'r-', label='验证准确率', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('准确率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. ROUGE分数曲线
        plt.subplot(2, 2, 4)
        rouge1_scores = [score['rouge1'] for score in self.rouge_scores]
        rouge2_scores = [score['rouge2'] for score in self.rouge_scores]
        rougeL_scores = [score['rougeL'] for score in self.rouge_scores]

        plt.plot(epochs, rouge1_scores, 'g-', label='ROUGE-1', linewidth=2)
        plt.plot(epochs, rouge2_scores, 'b-', label='ROUGE-2', linewidth=2)
        plt.plot(epochs, rougeL_scores, 'r-', label='ROUGE-L', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE Score')
        plt.title('ROUGE分数曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        image_filename = f'training_curves_{self.config.num_epochs}epochs_{self.config.n_heads}heads.png'
        image_path = os.path.join(self.config.image_dir, image_filename)
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 打印最终统计
        print(f"\n📊 最终统计:")
        print(f"最终训练损失: {self.train_losses[-1]:.4f}")
        print(f"最终验证损失: {self.val_losses[-1]:.4f}")
        print(f"最终训练困惑度: {self.train_perplexities[-1]:.2f}")
        print(f"最终验证困惑度: {self.val_perplexities[-1]:.2f}")
        print(f"最终训练准确率: {self.train_accuracies[-1]:.4f}")
        print(f"最终验证准确率: {self.val_accuracies[-1]:.4f}")
        if self.rouge_scores:
            final_rouge = self.rouge_scores[-1]
            print(f"最终ROUGE-1: {final_rouge['rouge1']:.4f}")
            print(f"最终ROUGE-2: {final_rouge['rouge2']:.4f}")
            print(f"最终ROUGE-L: {final_rouge['rougeL']:.4f}")


def main():
    """
    主函数
    """
    # 设置随机种子
    set_seed(config.seed)
    print("设置随机种子完成")

    # 获取数据
    print("加载数据...")
    train_loader, val_loader,test_loader, tokenizer = get_data_loaders(config)

    # 创建模型
    print("初始化模型...")
    # model = Transformer(config)
    model = Transformer(config, tokenizer=tokenizer)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数数量: {total_params:,}")

    # 测试不同注意力头数量的差别,改为8，要测试就把下面这行取消注释
    # config.n_heads = 8

    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader,test_loader, tokenizer, config)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()