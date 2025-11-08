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

        # 条件位置编码
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
    def __init__(self, config, tokenizer, train_loader, val_loader, test_loader):
        """
        初始化消融实验
        Args:
            config: 实验配置
            tokenizer: 分词器实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        self.config = config
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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

    def calculate_perplexity(self, loss):
        """计算困惑度"""
        return math.exp(loss)

    def calculate_accuracy(self, logits, labels, ignore_index=-100):
        """计算准确率"""
        predictions = torch.argmax(logits, dim=-1)
        valid_mask = (labels != ignore_index)
        correct = (predictions == labels) & valid_mask
        correct_count = correct.sum().item()
        total_valid = valid_mask.sum().item()

        if total_valid == 0:
            return 0.0
        accuracy = correct_count / total_valid
        return accuracy

    def train_model(self, use_positional_encoding=True, model_name="baseline"):
        """
        训练单个模型
        Args:
            use_positional_encoding: 是否使用位置编码
            model_name: 名称标识
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

        # 训练记录
        train_losses = []
        val_losses = []
        train_perplexities = []
        val_perplexities = []
        train_accuracies = []
        val_accuracies = []
        rouge_scores = []

        # 基于准确率选择最佳模型
        best_accuracy = 0.0
        best_model_state = None
        best_epoch = 0

        # 训练循环
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # 训练阶段
            model.train()
            total_train_loss = 0
            total_train_perplexity = 0
            total_train_accuracy = 0

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

                # 计算准确率
                accuracy = self.calculate_accuracy(
                    logits,
                    labels[:, 1:],
                    ignore_index=self.tokenizer.pad_token_id
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                total_train_loss += loss.item()
                total_train_perplexity += self.calculate_perplexity(loss.item())
                total_train_accuracy += accuracy

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'ppl': f'{self.calculate_perplexity(loss.item()):.2f}'
                })

            # 计算训练平均值
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_train_perplexity = total_train_perplexity / len(self.train_loader)
            avg_train_accuracy = total_train_accuracy / len(self.train_loader)

            train_losses.append(avg_train_loss)
            train_perplexities.append(avg_train_perplexity)
            train_accuracies.append(avg_train_accuracy)

            # 验证阶段
            avg_val_loss, avg_val_perplexity, avg_val_accuracy, rouge = self.validate_model(
                model, self.val_loader, criterion
            )

            val_losses.append(avg_val_loss)
            val_perplexities.append(avg_val_perplexity)
            val_accuracies.append(avg_val_accuracy)
            rouge_scores.append(rouge)

            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

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
        """验证模型性能"""
        model.eval()
        total_loss = 0
        total_perplexity = 0
        total_accuracy = 0
        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='验证'):
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

                # 计算准确率
                accuracy = self.calculate_accuracy(
                    logits,
                    labels[:, 1:],
                    ignore_index=self.tokenizer.pad_token_id
                )

                total_loss += loss.item()
                total_perplexity += self.calculate_perplexity(loss.item())
                total_accuracy += accuracy

                # 生成预测
                try:
                    src_mask = model._create_src_mask(input_ids)
                    predictions = model.generate(
                        input_ids,
                        src_mask=src_mask,
                        max_length=self.config.max_target_length
                    )

                    decoded_preds = self.tokenizer.batch_decode(predictions.cpu(), skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

                    all_predictions.extend(decoded_preds)
                    all_references.extend(decoded_labels)
                except Exception as e:
                    print(f"生成预测时出错: {e}")
                    continue

        # 计算平均指标
        avg_loss = total_loss / len(val_loader)
        avg_perplexity = total_perplexity / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)

        # 计算ROUGE分数
        if len(all_predictions) > 0 and len(all_references) > 0:
            rouge = self.calculate_rouge(all_predictions, all_references)
        else:
            rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        return avg_loss, avg_perplexity, avg_accuracy, rouge

    def calculate_rouge(self, predictions, references):
        """计算ROUGE分数"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
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
        ablation_results = self.train_model(
            use_positional_encoding=False,
            model_name="without_positional_encoding"
        )
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
        """保存实验结果"""
        # 保存JSON结果
        results_file = os.path.join(self.results_dir,
                                    f"ablation_results_{self.config.num_epochs}epochs_{self.config.n_heads}heads.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # 保存文本摘要
        summary_file = os.path.join(self.results_dir,
                                    f"experiment_summary_{self.config.num_epochs}epochs_{self.config.n_heads}heads.txt")
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

            f.write("性能差异（基线 - 消融）:\n")
            f.write(f"  验证准确率差异: {accuracy_diff:+.4f} ({accuracy_diff / baseline['final_val_accuracy']:+.1%})\n")
            f.write(f"  ROUGE-1差异: {rouge1_diff:+.4f} ({rouge1_diff / baseline['final_rouge']['rouge1']:+.1%})\n")
            f.write(f"  ROUGE-2差异: {rouge2_diff:+.4f} ({rouge2_diff / baseline['final_rouge']['rouge2']:+.1%})\n")
            f.write(f"  ROUGE-L差异: {rougeL_diff:+.4f} ({rougeL_diff / baseline['final_rouge']['rougeL']:+.1%})\n")

        print(f"✅ 实验结果已保存到: {results_file}")
        print(f"✅ 实验摘要已保存到: {summary_file}")

    def visualize_comparison(self):
        """可视化比较结果"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        baseline = self.results["with_positional_encoding"]
        ablation = self.results["without_positional_encoding"]

        epochs = range(1, len(baseline['train_losses']) + 1)

        # 1. 损失曲线比较
        axes[0, 0].plot(epochs, baseline['train_losses'], 'b-', label='有位置编码', linewidth=2)
        axes[0, 0].plot(epochs, ablation['train_losses'], 'r-', label='无位置编码', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('训练损失')
        axes[0, 0].set_title('训练损失比较')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 准确率曲线比较
        axes[0, 1].plot(epochs, baseline['val_accuracies'], 'b-', label='有位置编码', linewidth=2)
        axes[0, 1].plot(epochs, ablation['val_accuracies'], 'r-', label='无位置编码', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('验证准确率')
        axes[0, 1].set_title('验证准确率比较')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 困惑度曲线比较
        axes[1, 0].plot(epochs, baseline['val_perplexities'], 'b-', label='有位置编码', linewidth=2)
        axes[1, 0].plot(epochs, ablation['val_perplexities'], 'r-', label='无位置编码', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('验证困惑度')
        axes[1, 0].set_title('验证困惑度比较')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 最终性能柱状图
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