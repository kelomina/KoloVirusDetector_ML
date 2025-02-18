# KoloVirusDetector_ML

# 恶意软件检测模型训练工具

## 项目简介
本项目是一个基于机器学习的恶意软件检测系统，通过分析PE文件特征，使用集成学习算法实现对恶意软件的识别。系统支持增量训练和模型微调，能够有效处理样本不平衡问题，并具备特征自动压缩与选择功能。

## 主要功能
- **多维度特征提取**
  - PE文件结构分析
  - 熵值计算
  - 数字签名验证
  - 图标特征提取
  - 字符串特征编码
  - 导入/导出表分析

- **智能训练流程**
  - 自动化特征工程（标准化/PCA压缩）
  - 集成Stacking模型（LightGBM + 随机森林 + 逻辑回归）
  - 超参数自动优化
  - 动态特征重要性筛选
  - 数据增强（ADASYN）

- **生产级特性**
  - 多进程并行处理
  - 增量训练支持
  - 模型/特征持久化
  - 全流程日志追踪
  - 超时容错机制

## 系统要求
- Python 3.8+
- Windows/Linux
- 推荐配置：
  - CPU: 4核以上
  - GPU：RTX690或以上（Nvidia显卡优先）
  - 内存: 16GB+
  - 存储: SSD优先

## 快速开始

### 安装依赖
CUDA12.8(非Nvidia的GPU勿装)[https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe]
```bash
pip install -r requirements.txt
```

### 配置文件示例（config.json）
```json
{
    "virus_samples_dir": "path/to/virus_samples",
    "benign_samples_dir": "path/to/benign_samples",
    "features_path": "features.npy",
    "labels_path": "labels.npy",
    "model_output_path": "malware_detector.joblib"
}
```

### 训练命令
```bash
# 增量训练模式
python train.py --config config.json --mode incremental

# 微调模式（需已有基础模型）
python train.py --config config.json --mode fine_tune
```

## 目录结构
```
.
├── train.py                 # 主训练脚本
├── config.json              # 配置文件示例
├── requirements.txt         # 依赖列表
├── model.joblib             # 训练好的模型
├── features.npy             # 特征存储文件
├── labels.npy               # 标签存储文件
└── train_virus_detector.log # 训练日志
```

## 关键参数说明
| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（必须） |
| `--mode` | 训练模式：`incremental`(增量)/`fine_tune`(微调) |
| `virus_samples_dir` | 恶意样本目录 |
| `benign_samples_dir` | 良性样本目录 |
| `batch_size` | 文件处理批大小（默认512） |
| `max_workers` | 最大并发进程数（默认4） |

## 注意事项
1. 样本目录要求：
   - 病毒样本应为PE格式的可执行文件
   - 良性样本需经过人工验证
   - 建议样本量：每类不少于1000个

2. 特征工程：
   - 自动处理不同长度特征向量
   - 支持95%方差保留的PCA压缩
   - 特征标准化预处理

3. 性能优化：
   - 使用CUDA进行训练加速
   - 使用内存映射文件加速读取
   - 支持多进程并行特征提取
   - 自动跳过无效文件

## 模型特性
| 指标 | 说明 |
|------|------|
| 架构 | Stacking集成模型 |
| 基础模型 | LightGBM + 随机森林 + 逻辑回归 |
| 优化方式 | RandomizedSearchCV超参数搜索 |
| 特征维度 | 动态调整（50-1000） |
| 训练时间 | 约1小时/万样本（i7-12700H） |

## 常见问题
**Q：如何处理新样本？**
- 将新样本放入对应目录后，使用`incremental`模式进行增量训练

**Q：如何调整模型敏感度？**
- 修改`feature_importance_threshold`参数控制特征筛选强度

**Q：遇到内存不足怎么办？**
- 降低`batch_size`和`max_workers`参数值
- 使用特征压缩（默认已启用）

**Q：如何处理非PE文件？**
- 系统会自动跳过无效文件并在日志中记录

## 许可协议
本项目采用 MIT License 开源，欢迎贡献代码和提出改进建议。
