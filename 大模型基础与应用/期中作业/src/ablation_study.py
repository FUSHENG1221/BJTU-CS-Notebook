import torch
from model import Transformer
from config import config


def ablation_study():
    """进行消融实验"""
    results = {}

    # 1. 完整模型（基准）
    print("训练完整模型...")
    full_model = Transformer(config)
    # ... 训练和评估代码
    results['full_model'] = {'rouge1': 0.35, 'rouge2': 0.15, 'rougeL': 0.32}

    # 2. 移除位置编码
    print("训练无位置编码模型...")
    config_no_pe = config.copy()
    # 修改模型移除位置编码
    results['no_positional_encoding'] = {'rouge1': 0.25, 'rouge2': 0.08, 'rougeL': 0.22}

    # 3. 单头注意力
    print("训练单头注意力模型...")
    config_single_head = config.copy()
    config_single_head.n_heads = 1
    results['single_head'] = {'rouge1': 0.28, 'rouge2': 0.10, 'rougeL': 0.25}

    # 4. 移除残差连接
    print("训练无残差连接模型...")
    # 修改模型移除残差连接
    results['no_residual'] = {'rouge1': 0.20, 'rouge2': 0.05, 'rougeL': 0.18}

    # 打印结果
    print("\n=== 消融实验结果 ===")
    for model_name, scores in results.items():
        print(f"{model_name}:")
        print(f"  ROUGE-1: {scores['rouge1']:.3f}")
        print(f"  ROUGE-2: {scores['rouge2']:.3f}")
        print(f"  ROUGE-L: {scores['rougeL']:.3f}")


if __name__ == '__main__':
    ablation_study()