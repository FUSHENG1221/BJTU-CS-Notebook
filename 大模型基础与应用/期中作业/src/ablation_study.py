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

from model import Transformer
from config import config
from data_loader import get_data_loaders
from utils import set_seed

import warnings
warnings.filterwarnings("ignore")# 忽略警告输出
plt.rcParams["font.sans-serif"] = ["SimHei"]# 设置matplotlib中文字体显示（解决中文乱码问题）
plt.rcParams["axes.unicode_minus"] = False# 显示负号

class AblationTransformer(Transformer):
    """
    消融实验专用Transformer模型
    支持启用/禁用位置编码功能
    """
    def __init__(self, config, tokenizer=None, use_positional_encoding=True):
        """
        初始化消融实验模型
        Args:
            config: 模型配置
            tokenizer: 分词器
            use_positional_encoding: 是否使用位置编码（消融变量）
        """
        super().__init__(config, tokenizer)
        self.use_positional_encoding = use_positional_encoding

        # 如果禁用位置编码，将位置编码层替换为恒等映射
        if not use_positional_encoding:
            self.pos_encoding = nn.Identity()  # 恒等映射，不添加位置信息

    def encode(self, src, src_mask=None):
        """重写编码方法，支持位置编码开关"""
        if src_mask is None:
            src_mask = self._create_src_mask(src)

        # 词嵌入
        x = self.embedding(src)

        # 条件位置编码，根据`是否使用位置编码（消融变量）`决定
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        encoder_self_attentions = []
        for layer in self.encoder_layers:
            x, self_attn = layer(x, src_mask)
            encoder_self_attentions.append(self_attn)

        return x, encoder_self_attentions

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """重写解码方法，支持位置编码开关"""
        if tgt_mask is None:
            tgt_mask = self._create_tgt_mask(tgt.size(1))

        # 词嵌入
        x = self.embedding(tgt)

        # 条件位置编码
        if self.use_positional_encoding:
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


class AblationStudy:
    """
    位置编码消融实验类
    比较使用位置编码 vs 不使用位置编码的模型性能差异
    """

    def __init__(self, config, tokenizer,train_loader, val_loader, test_loader):
        """
        初始化消融实验

        Args:
            config: 实验配置
            tokenizer: 分词器实例
        """
        self.config = config
        self.tokenizer = tokenizer
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.device = config.device

        # 创建结果目录
        self.results_dir = "../results/ablation_study"
        os.makedirs(self.results_dir, exist_ok=True)

        # 实验记录
        self.results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "config": config.__dict__,
                "description": "位置编码消融实验：比较使用位置编码 vs 不使用位置编码的模型性能"
            },
            "with_positional_encoding": {},
            "without_positional_encoding": {}
        }

    def train_model(self, use_positional_encoding=True, model_name="baseline"):
        """
        训练单个模型
        Args:
            use_positional_encoding: 是否使用位置编码
            model_name: 模型名称标识
        Returns:
            dict: 训练结果和模型状态
        """
        print(f"\n{'=' * 60}")
        print(f"训练模型: {model_name}")
        print(f"使用位置编码: {use_positional_encoding}")
        print(f"{'=' * 60}")

        # 设置随机种子确保可重复性
        set_seed(self.config.seed)

        # 创建模型
        model = AblationTransformer(
            self.config,
            self.tokenizer,
            use_positional_encoding=use_positional_encoding
        ).to(self.device)

        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )

        # 损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # 数据加载器
        # train_loader, val_loader, test_loader, tokenizer = get_data_loaders(self.config)

        # 训练记录
        train_losses = []
        val_losses = []
        rouge_scores = []
        best_rouge = 0
        best_model_state = None

        # 训练循环
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # 训练阶段
            model.train()
            total_train_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'训练 {model_name}')

            for batch in progress_bar:
                # 准备数据
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = model(input_ids, labels[:, :-1])
                logits = outputs['logits']

                # 计算损失
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1)
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_train_loss = total_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            avg_val_loss, rouge = self.validate_model(model, self.val_loader, criterion)
            val_losses.append(avg_val_loss)
            rouge_scores.append(rouge)

            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # 打印结果
            print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
            print(f"学习率: {current_lr:.2e}")
            print(f"ROUGE-1: {rouge['rouge1']:.4f}, ROUGE-2: {rouge['rouge2']:.4f}, ROUGE-L: {rouge['rougeL']:.4f}")

            # 保存最佳模型
            if rouge['rouge1'] > best_rouge:
                best_rouge = rouge['rouge1']
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'rouge': rouge
                }

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'rouge_scores': rouge_scores,
            'best_rouge': best_rouge,
            'best_model_state': best_model_state
        }

    def validate_model(self, model, val_loader, criterion):
        """验证模型性能"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='验证'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 计算损失
                outputs = model(input_ids, labels[:, :-1])
                logits = outputs['logits']
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1)
                )
                total_loss += loss.item()

                # 生成预测
                try:
                    predictions = model.generate(input_ids, max_length=self.config.max_target_length)
                    decoded_preds = self.tokenizer.batch_decode(predictions.cpu(), skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

                    all_predictions.extend(decoded_preds)
                    all_references.extend(decoded_labels)
                except Exception as e:
                    print(f"生成预测时出错: {e}")
                    continue

        # 计算ROUGE分数
        if len(all_predictions) > 0:
            rouge = self.calculate_rouge(all_predictions, all_references)
        else:
            rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        avg_loss = total_loss / len(val_loader)
        return avg_loss, rouge
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
    def run_study(self):
        """运行完整的消融实验"""
        print("开始位置编码消融实验")
        print("=" * 60)

        # 实验1: 使用位置编码（基线模型）
        print("\n🔬 实验1: 训练带有位置编码的模型（基线）")
        baseline_results = self.train_model(
            use_positional_encoding=True,
            model_name="with_positional_encoding"
        )
        self.results["with_positional_encoding"] = {
            "final_train_loss": baseline_results['train_losses'][-1],
            "final_val_loss": baseline_results['val_losses'][-1],
            "final_rouge": baseline_results['rouge_scores'][-1],
            "best_rouge": baseline_results['best_rouge'],
            "all_rouge_scores": baseline_results['rouge_scores'],
            "train_losses": baseline_results['train_losses'],
            "val_losses": baseline_results['val_losses']
        }

        # 实验2: 不使用位置编码（消融模型）
        print("\n🔬 实验2: 训练不带有位置编码的模型（消融）")
        ablation_results = self.train_model(
            use_positional_encoding=False,
            model_name="without_positional_encoding"
        )
        self.results["without_positional_encoding"] = {
            "final_train_loss": ablation_results['train_losses'][-1],
            "final_val_loss": ablation_results['val_losses'][-1],
            "final_rouge": ablation_results['rouge_scores'][-1],
            "best_rouge": ablation_results['best_rouge'],
            "all_rouge_scores": ablation_results['rouge_scores'],
            "train_losses": ablation_results['train_losses'],
            "val_losses": ablation_results['val_losses']
        }

        # 保存结果
        self.save_results()

        # 可视化比较
        self.visualize_comparison()

        # 分析结果
        self.analyze_results()

        return self.results

    def save_results(self):
        """保存实验结果"""
        # 保存JSON结果
        results_file = os.path.join(self.results_dir, f"ablation_results_{self.config.num_epochs}epochs_{self.config.n_heads}heads.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # 保存文本摘要
        summary_file = os.path.join(self.results_dir, f"experiment_summary_{self.config.num_epochs}epochs_{self.config.n_heads}heads.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("位置编码消融实验摘要\n")
            f.write("=" * 50 + "\n\n")

            baseline = self.results["with_positional_encoding"]
            ablation = self.results["without_positional_encoding"]

            f.write("基线模型（有位置编码）:\n")
            f.write(f"  最终训练损失: {baseline['final_train_loss']:.4f}\n")
            f.write(f"  最终验证损失: {baseline['final_val_loss']:.4f}\n")
            f.write(f"  最终ROUGE-1: {baseline['final_rouge']['rouge1']:.4f}\n")
            f.write(f"  最佳ROUGE-1: {baseline['best_rouge']:.4f}\n\n")

            f.write("消融模型（无位置编码）:\n")
            f.write(f"  最终训练损失: {ablation['final_train_loss']:.4f}\n")
            f.write(f"  最终验证损失: {ablation['final_val_loss']:.4f}\n")
            f.write(f"  最终ROUGE-1: {ablation['final_rouge']['rouge1']:.4f}\n")
            f.write(f"  最佳ROUGE-1: {ablation['best_rouge']:.4f}\n\n")

            # 计算差异
            rouge1_diff = baseline['final_rouge']['rouge1'] - ablation['final_rouge']['rouge1']
            rouge2_diff = baseline['final_rouge']['rouge2'] - ablation['final_rouge']['rouge2']
            rougeL_diff = baseline['final_rouge']['rougeL'] - ablation['final_rouge']['rougeL']

            f.write("性能差异（基线 - 消融）:\n")
            f.write(f"  ROUGE-1差异: {rouge1_diff:+.4f}\n")
            f.write(f"  ROUGE-2差异: {rouge2_diff:+.4f}\n")
            f.write(f"  ROUGE-L差异: {rougeL_diff:+.4f}\n")

        print(f"✅ 实验结果已保存到: {results_file}")
        print(f"✅ 实验摘要已保存到: {summary_file}")

    def visualize_comparison(self):
        """可视化比较结果"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        baseline = self.results["with_positional_encoding"]
        ablation = self.results["without_positional_encoding"]

        epochs = range(1, len(baseline['train_losses']) + 1)

        # 1. 训练损失比较
        axes[0, 0].plot(epochs, baseline['train_losses'], 'b-', label='有位置编码', linewidth=2)
        axes[0, 0].plot(epochs, ablation['train_losses'], 'r-', label='无位置编码', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('训练损失')
        axes[0, 0].set_title('训练损失比较')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 验证损失比较
        axes[0, 1].plot(epochs, baseline['val_losses'], 'b-', label='有位置编码', linewidth=2)
        axes[0, 1].plot(epochs, ablation['val_losses'], 'r-', label='无位置编码', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('验证损失')
        axes[0, 1].set_title('验证损失比较')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. ROUGE-1分数比较
        baseline_rouge1 = [score['rouge1'] for score in baseline['all_rouge_scores']]
        ablation_rouge1 = [score['rouge1'] for score in ablation['all_rouge_scores']]
        axes[1, 0].plot(epochs, baseline_rouge1, 'b-', label='有位置编码', linewidth=2)
        axes[1, 0].plot(epochs, ablation_rouge1, 'r-', label='无位置编码', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ROUGE-1分数')
        axes[1, 0].set_title('ROUGE-1分数比较')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 最终性能柱状图
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        baseline_scores = [
            baseline['final_rouge']['rouge1'],
            baseline['final_rouge']['rouge2'],
            baseline['final_rouge']['rougeL']
        ]
        ablation_scores = [
            ablation['final_rouge']['rouge1'],
            ablation['final_rouge']['rouge2'],
            ablation['final_rouge']['rougeL']
        ]

        x = np.arange(len(metrics))
        width = 0.35
        axes[1, 1].bar(x - width / 2, baseline_scores, width, label='有位置编码', alpha=0.8)
        axes[1, 1].bar(x + width / 2, ablation_scores, width, label='无位置编码', alpha=0.8)
        axes[1, 1].set_xlabel('评估指标')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].set_title('最终性能比较')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, f"ablation_comparison_{self.config.num_epochs}epochs_{self.config.n_heads}heads.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 比较图表已保存到: {plot_file}")

    def analyze_results(self):
        """分析实验结果并生成见解"""
        baseline = self.results["with_positional_encoding"]
        ablation = self.results["without_positional_encoding"]

        print("\n" + "=" * 60)
        print("🔍 位置编码消融实验分析结果")
        print("=" * 60)

        # 计算性能差异
        rouge1_diff = baseline['final_rouge']['rouge1'] - ablation['final_rouge']['rouge1']
        rouge2_diff = baseline['final_rouge']['rouge2'] - ablation['final_rouge']['rouge2']
        rougeL_diff = baseline['final_rouge']['rougeL'] - ablation['final_rouge']['rougeL']

        # 训练稳定性分析
        baseline_loss_std = np.std(baseline['train_losses'][-5:])  # 最后5个epoch的标准差
        ablation_loss_std = np.std(ablation['train_losses'][-5:])

        print(f"\n📊 性能比较:")
        print(f"ROUGE-1差异: {rouge1_diff:+.4f} ({rouge1_diff / baseline['final_rouge']['rouge1']:+.1%})")
        print(f"ROUGE-2差异: {rouge2_diff:+.4f} ({rouge2_diff / baseline['final_rouge']['rouge2']:+.1%})")
        print(f"ROUGE-L差异: {rougeL_diff:+.4f} ({rougeL_diff / baseline['final_rouge']['rougeL']:+.1%})")

        print(f"\n📈 训练稳定性:")
        print(f"基线模型损失标准差: {baseline_loss_std:.4f}")
        print(f"消融模型损失标准差: {ablation_loss_std:.4f}")

        # 生成见解
        print(f"\n💡 实验发现与见解:")

        if rouge1_diff > 0.1:
            print("1. 🎯 位置编码对模型性能有显著影响")
            print("   - 位置编码提供了关键的序列顺序信息")
            print("   - 对于摘要生成任务，理解对话顺序至关重要")
        elif rouge1_diff > 0.05:
            print("1. ⚠️ 位置编码对模型性能有中等影响")
            print("   - 位置信息有助于但不完全决定模型性能")
            print("   - 模型可能从内容中学习到部分顺序信息")
        else:
            print("1. 🔄 位置编码对模型性能影响较小")
            print("   - 模型可能主要依赖内容信息而非顺序信息")
            print("   - 或者模型通过其他方式学习到了顺序关系")

        if ablation_loss_std > baseline_loss_std * 1.5:
            print("2. 📉 无位置编码时训练更不稳定")
            print("   - 位置编码有助于训练收敛和稳定性")
            print("   - 缺乏位置信息可能导致优化困难")
        else:
            print("2. 📊 训练稳定性相当")
            print("   - 位置编码对训练稳定性影响有限")

        print("3. 🔬 技术见解:")
        print("   - Transformer的自注意力机制本身不具备位置感知能力")
        print("   - 位置编码是序列顺序信息的唯一来源")
        print("   - 在对话摘要任务中，时间顺序对理解对话流程很重要")

        print("4. 🚀 实践建议:")
        if rouge1_diff > 0.05:
            print("   - 在类似任务中必须使用位置编码")
            print("   - 可以尝试更复杂的位置编码方案（如相对位置编码）")
        else:
            print("   - 对于某些任务，可以简化模型架构")
            print("   - 但仍建议使用位置编码以确保最佳性能")


def main():
    """主函数：运行消融实验"""
    # 设置随机种子
    set_seed(config.seed)
    print("设置随机种子完成")

    # 为消融实验调整配置（加快实验速度）
    config.num_epochs = 2  # 减少训练轮数进行快速实验
    config.batch_size = 16  # 减小批次大小

    # 获取数据
    print("加载数据...")
    train_loader, val_loader, test_loader, tokenizer = get_data_loaders(config)

    # 运行消融实验
    study = AblationStudy(config, tokenizer,train_loader, val_loader, test_loader)
    results = study.run_study()

    print("\n🎉 消融实验完成！")
    print(f"结果保存在: {study.results_dir}")


if __name__ == "__main__":
    main()