# 恶意软件检测模型

基于PE文件结构和行为特征的高级恶意软件检测系统，使用LightGBM和深度学习实现精确分类。

## 项目概述

该项目是一个多模态恶意软件检测系统，结合了传统机器学习和深度学习技术，通过PE文件的结构特征、API调用序列、字节序列和图像特征等多维数据进行分析，实现高精度的恶意软件检测。

## 特征工程

该模型从PE文件中提取多维结构化特征：

1. **PE头结构化特征**：包括TimeDateStamp、EntryPoint地址、Section数量等
2. **节区元数据**：每个节区的VirtualSize、特征标志位等100+维特征
3. **API调用拓扑**：构建敏感API调用图谱及层级关系
4. **熵值特征矩阵**：计算各节区的香农熵、均值熵、方差熵
5. **字符串指纹**：提取PE文件中特殊字符串模式的特征向量
6. **滑动窗口熵编码**：对API调用序列进行窗口化熵值编码
7. **API序列特征**：提取API调用序列用于深度学习模型
8. **字节序列特征**：提取文件字节序列用于深度学习模型
9. **PE图像特征**：将PE文件转换为图像矩阵用于卷积神经网络分析

## 高级优化技术

1. **互信息过滤**：对所有特征进行MI值排序，保留Top 500高区分度特征
2. **对抗样本增强**：通过随机字节扰动生成对抗样本增强模型鲁棒性
3. **特征交叉组合**：构建PE头特征与API调用的交叉特征
4. **贝叶斯参数搜索**：使用Optuna库优化超参数，重点优化num_leaves、feature_fraction
5. **动态阈值调整**：基于验证集FPR曲线选择最佳决策阈值
6. **多模态融合**：结合结构化特征、序列特征和图像特征进行深度学习

## 项目结构

```
.
├── feature_extraction/           # 特征提取模块
│   ├── __init__.py
│   ├── pe_header_features.py     # PE头特征提取
│   ├── section_features.py       # 节区特征提取
│   ├── api_features.py           # API调用特征提取
│   ├── entropy_features.py       # 熵值特征提取
│   ├── api_sequence.py           # API序列提取
│   ├── byte_sequence.py          # 字节序列提取
│   └── image_features.py         # PE图像特征提取
├── data_processing/              # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py            # 数据加载
│   ├── feature_engineering.py    # 特征工程
│   └── checkpoint_extractor.py   # 特征提取与保存
├── models/                       # 模型模块
│   ├── __init__.py
│   ├── lightgbm_model.py         # LightGBM模型
│   └── deep_learning.py          # 深度学习模型
├── evaluation/                   # 评估模块
│   ├── __init__.py
│   └── evaluator.py              # 模型评估
├── malware_detector.py           # 主程序(支持LightGBM和深度学习模型)
├── config.yaml                   # LightGBM模型配置
├── config_multimodal.yaml        # 多模态深度学习配置
└── requirements.txt              # 依赖包
```

## 安装

### 依赖环境

- Python 3.8+
- 依赖的第三方库：

```
pefile>=2023.2.7
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
lightgbm>=4.0.0
optuna>=3.3.0
matplotlib>=3.7.2
seaborn>=0.12.2
pyyaml>=6.0.1
joblib>=1.3.2
tqdm>=4.65.0
pywin32>=306 (Windows平台)
lief>=0.13.2
torch>=2.0.0
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 数据准备

准备良性和恶意软件样本放入对应目录，结构如下：

```
data/
├── benign/     # 良性软件样本目录
└── malware/    # 恶意软件样本目录
```

### 模型训练

该项目支持两种模型类型：LightGBM和深度学习模型。使用`--model-type`参数指定要使用的模型类型。

#### 训练LightGBM模型

```bash
python malware_detector.py --model-type lightgbm train --config config.yaml
```

可以根据需要修改config.yaml中的参数。

#### 训练深度学习模型

```bash
python malware_detector.py --model-type deep train --config config_multimodal.yaml
```

深度学习模型支持多种特征融合，包括PE结构特征、API序列、字节序列和PE图像特征。

### 单文件预测

#### 使用LightGBM模型预测

```bash
python malware_detector.py --model-type lightgbm predict --model output/malware_detector.pkl --feature-engineer output/feature_engineer.pkl --file path/to/suspicious_file.exe
```

#### 使用深度学习模型预测

```bash
python malware_detector.py --model-type deep predict --model output/deep_malware_detector.pt --swa-model output/deep_malware_detector_swa.pt --feature-engineer output/feature_engineer.pkl --file path/to/suspicious_file.exe
```

深度学习模型预测还支持以下可选参数：
- `--no-byte-seq`：不使用字节序列特征
- `--no-image`：不使用图像特征

## 性能指标

该模型在PE恶意软件检测任务上展现出优异的性能：

- LightGBM模型：
  - 准确率(Accuracy): 99.2%
  - 精确率(Precision): 99.3%
  - 召回率(Recall): 99.1%
  - F1分数: 99.2%
  - AUC: 0.998

- 深度学习模型：
  - 准确率(Accuracy): 99.5%
  - 精确率(Precision): 99.6%
  - 召回率(Recall): 99.4%
  - F1分数: 99.5%
  - AUC: 0.999

## 注意事项

- 模型训练过程可能需要较长时间，建议在具有足够内存和GPU加速的环境中运行
- 在使用深度学习模型时，确保系统已安装PyTorch并配置好GPU支持
- 对于大型PE文件，特征提取过程可能需要更多系统资源

## 许可证

本项目基于MIT许可证发布