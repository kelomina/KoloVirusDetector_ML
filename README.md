# KoloVirusDetector_ML

# 恶意软件检测系统

基于PE文件特征和机器学习算法的恶意软件检测系统，能够有效识别病毒文件与良性文件。

## 项目亮点

- **多线程特征提取**：支持批量文件处理，提升特征提取效率
- **专业PE文件解析**：深入分析PE头结构、节区特征、导入函数等关键信息
- **智能数据增强**：集成ADASYN/SMOTE算法处理类别不平衡
- **模型自动调优**：基于GridSearchCV的自动超参数优化
- **增量学习支持**：支持增量训练和模型微调两种模式
- **生产级日志系统**：完整记录训练过程和关键指标

## 环境要求

- Python 3.6+
- 依赖库：
  ```
  lightgbm==3.3.5
  pefile==2023.2.7
  numpy==1.23.5
  scikit-learn==1.2.2
  imbalanced-learn==0.10.1
  tqdm==4.65.0
  ```

## 快速开始

### 1. 安装依赖
```bash
pip install lightgbm pefile numpy scikit-learn imbalanced-learn tqdm
```

### 2. 准备数据
```
样本库/
├── 待拉黑/        # 病毒样本
└── 待加入白名单/   # 良性样本
```

### 3. 配置文件示例（config.json）
```json
{
    "virus_samples_dir": "E:\\样本库\\待拉黑",
    "benign_samples_dir": "E:\\样本库\\待加入白名单",
    "features_path": "features.npy",
    "labels_path": "labels.npy",
    "model_output_path": "malware_detector.joblib"
}
```

### 4. 启动训练
```bash
# 增量训练模式（推荐首次训练）
python train.py --config config.json --mode incremental

# 微调模式（已有模型基础上优化）
python train.py --config config.json --mode fine_tune
```

## 项目结构
```
malware-detector/
├── train.py               # 主训练程序
├── config.json            # 配置文件示例
├── features.npy           # 特征数据存储
├── labels.npy             # 标签数据存储
├── model.joblib           # 训练好的模型
└── train_virus_detector.log  # 训练日志
```

## 配置说明
| 参数                 | 说明                          | 示例值                          |
|----------------------|-----------------------------|--------------------------------|
| virus_samples_dir    | 病毒样本目录                    | "E:\\样本库\\待拉黑"           |
| benign_samples_dir   | 良性样本目录                    | "E:\\样本库\\待加入白名单"     |
| features_path        | 特征数据保存路径                | "features.npy"                |
| labels_path          | 标签数据保存路径                | "labels.npy"                  |
| model_output_path    | 模型保存路径                    | "malware_detector.joblib"     |

## 特征工程
系统提取超过600维特征，包括：
1. **基础特征**
   - 文件熵值
   - 字节分布直方图
   - 首尾128字节原始数据

2. **PE结构特征**
   - DOS头/文件头/可选头关键字段
   - 节区信息（.text段熵值、.data段大小）
   - 导入函数特征（排序后前32个函数）

3. **元数据特征**
   - 文件描述信息
   - 版权信息
   - 版本信息

## 训练结果示例
```
最佳模型: LGBMClassifier
最佳准确率: 0.98

              precision    recall  f1-score   support

           0       0.97      0.99      0.98      1234
           1       0.99      0.97      0.98      1156

    accuracy                           0.98      2390
   macro avg       0.98      0.98      0.98      2390
weighted avg       0.98      0.98      0.98      2390
```

## 注意事项
1. 样本安全：建议在隔离的虚拟环境中运行
2. 硬件要求：建议配备至少16GB内存
3. 数据平衡：初始样本建议保持1:1~1:3的病毒/良性比例
4. 模型更新：推荐每月进行增量训练保持检测能力

## 许可协议
本项目采用 MIT 开源许可证，详见 LICENSE 文件。

## 致谢
项目基于以下开源技术构建：
- LightGBM by Microsoft
- pefile by Ero Carrera
- scikit-learn 社区

欢迎提交 Issue 和 PR 共同改进本项目！
