import torch
import numpy as np
import random

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_attention_patterns(model, tokenizer, text):
    """分析注意力模式"""
    model.eval()
    with torch.no_grad():
        # 编码输入
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # 前向传播
        outputs = model(input_ids, input_ids)
        
        # 分析注意力权重
        attention_weights = outputs['encoder_self_attentions']
        
        return attention_weights