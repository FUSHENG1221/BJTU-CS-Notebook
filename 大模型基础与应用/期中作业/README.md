# 大模型期中作业-万倩

本项目基于PyTorch从零实现了一个完整的Transformer模型，包含encoder和decoder结构，并在小规模文本数据集【Hugging Face 提供的SAMSum数据集，链接为：[knkarthick/samsum · Datasets at Hugging Face](https://huggingface.co/datasets/knkarthick/samsum)】上进行了语言建模任务的训练和验证。

## 项目结构

```
期中作业/
├── data/              # 存储数据集文件和分词器
│   ├── sasum         # 数据集
│   └── tokenizer              # 分词器
├── src/                    # 源代码目录
│   ├── model.py           # Transformer模型实现
│   ├── train.py         # 主要实验
│   ├── data_loader.py     # 数据加载和预处理
│   ├── config.py          # 配置文件
│   ├── ablation_study.py  # 消融实验
│   └── utils.py           # 工具函数
├── scripts/              # 配置文件目录
│   └── run.sh              # 主运行脚本
├── results/               # 实验结果
│   ├── ablation_study 		# 消融实验结果存储文件夹
│   ├── checkpoints 		# 中间模型存储文件夹
│   ├── images 				# 训练结果存储文件夹
│   └── model  				# 最佳模型和最后模型存储文件夹
├── requirements.txt       # 依赖包列表
└── README.md             # 说明文档
```

## 环境要求

### 硬件要求

- GPU: NVIDIA GPU with ≥ 8GB VRAM (推荐)
- 内存: ≥ 16GB RAM
- 存储: ≥ 5GB 可用空间

### 软件依赖

- Python 3.12
- PyTorch 2.9
- CUDA 12.6

## 安装步骤

1、创建conda环境：

```bash
conda create -n transformer python=3.12 -y
conda activate transformer
```

2、安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 训练模型

```bash
cd src
python train.py
```

### 多头注意力机制对比实验

```bash
# 取消train.py文件中“# config.n_heads = 8”的注释，再运行下面的命令
python train.py
```

### 消融实验（位置编码）

```bash
python abalation_study.py
```

### 结果查看验证

- 训练完成后，在`results/images`目录下查看训练模型和多头主机里机制对比实验的结果图
- 消融实验的结果存储在`results/ablation_study`目录下

## 配置文件说明

主要超参数设置：

| 参数          | 默认值 | 说明             |
| :------------ | :----- | :--------------- |
| d_model       | 256    | 模型维度         |
| num_heads     | 4      | 注意力头数       |
| num_layers    | 2      | Transformer层数  |
| ff_dim        | 512    | 前馈网络维度     |
| batch_size    | 16     | 批次大小         |
| learning_rate | 3e-4   | 学习率           |
| epochs        | 20     | 训练轮数         |
| dropout       | 0.1    | Dropout率        |
| warmup_steps  | 1000   | 学习率预热步数   |
| max_grad_norm | 1.0    | 最大梯度范数阈值 |
| seed          | 42     | 随机种子         |

## 功能特性

### 已实现模块

- ✅ Multi-head Self-Attention
- ✅ Position-wise Feed-Forward Network
- ✅ 残差连接 + Layer Normalization
- ✅ 正弦位置编码
- ✅ Encoder-Decoder架构
- ✅ 掩码机制（padding mask, future mask）

### 训练优化

- ✅ AdamW优化器
- ✅ 学习率调度（带预热）
- ✅ 梯度裁剪
- ✅ 模型保存/加载
- ✅ 训练曲线可视化

### 实验功能

- ✅ 消融实验支持
- ✅ 超参数敏感性分析

## 代码仓库地址

代码仓库地址为：[BJTU-CS-Notebook/大模型基础与应用/期中作业 at master · FUSHENG1221/BJTU-CS-Notebook](https://github.com/FUSHENG1221/BJTU-CS-Notebook/tree/master/大模型基础与应用/期中作业)
