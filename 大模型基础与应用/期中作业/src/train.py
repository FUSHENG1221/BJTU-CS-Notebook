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

class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        
        # 优化器和损失函数
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.rouge_scores = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in progress_bar:
            # 准备数据
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            labels = batch['labels'].to(self.config.device)
            
            # 前向传播（使用teacher forcing）
            outputs = self.model(input_ids, labels[:, :-1])
            logits = outputs['logits']
            
            # 计算损失
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                # 前向传播
                outputs = self.model(input_ids, labels[:, :-1])
                logits = outputs['logits']
                
                # 计算损失
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
                
                # 生成预测
                predictions = self.model.generate(input_ids)
                decoded_preds = self.tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                decoded_labels = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                all_predictions.extend(decoded_preds)
                all_references.extend(decoded_labels)
        
        # 计算ROUGE分数
        rouge = self.calculate_rouge(all_predictions, all_references)
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        self.rouge_scores.append(rouge)
        
        return avg_loss, rouge

    def calculate_rouge(self, predictions, references):
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

    def train(self):
        print("开始训练...")
        best_rouge = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, rouge_scores = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印结果
            print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            print(f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
            print(f'ROUGE-1: {rouge_scores["rouge1"]:.4f}, ROUGE-2: {rouge_scores["rouge2"]:.4f}, ROUGE-L: {rouge_scores["rougeL"]:.4f}')
            
            # 保存最佳模型
            if rouge_scores['rouge1'] > best_rouge:
                best_rouge = rouge_scores['rouge1']
                torch.save(self.model.state_dict(), 'best_model.pth')
        
        # 绘制训练曲线
        self.plot_training_curves()

    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练和验证损失')
        
        plt.subplot(1, 2, 2)
        rouge1_scores = [score['rouge1'] for score in self.rouge_scores]
        rouge2_scores = [score['rouge2'] for score in self.rouge_scores]
        rougeL_scores = [score['rougeL'] for score in self.rouge_scores]
        
        plt.plot(rouge1_scores, label='ROUGE-1')
        plt.plot(rouge2_scores, label='ROUGE-2')
        plt.plot(rougeL_scores, label='ROUGE-L')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE Score')
        plt.legend()
        plt.title('ROUGE分数变化')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

def main():
    # 设置随机种子
    set_seed(config.seed)
    
    # 获取数据
    train_loader, val_loader, tokenizer = get_data_loaders(config)
    
    # 创建模型
    model = Transformer(config)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()