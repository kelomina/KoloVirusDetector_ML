# KoloVirusDetector_ML

# 概述

本项目旨在训练一个恶意软件检测模型，使用多种机器学习算法（如随机森林、梯度提升、逻辑回归和 LightGBM）来区分恶意软件和良性软件。项目通过提取 PE 文件的特征，并使用这些特征来训练和评估模型。

## 安装指南

### 依赖安装

确保你已经安装了以下依赖项：

- `Python 3.6 或更高版本`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `pefile`
- `joblib`
- `imbalanced-learn`
- `tqdm`
- `argparse`
- `json`

你可以使用以下命令安装这些依赖项：

bash``````

##使用方法

###配置文件
首先，创建一个配置文件 config.json，示例如下：

json```
{
    "virus_samples_dir": "E:\\样本库\\待拉黑",
    "benign_samples_dir": "E:\\样本库\\待加入白名单",
    "features_path": "features.npy",
    "labels_path": "labels.npy",
    "model_path": "ML.pkl"
}```
运行脚本


使用以下命令运行训练脚本：

bash```
python train_virus_detector.py --config config.json```
##参数说明
--config: 配置文件路径，包含样本目录和输出文件路径。

输出
`features.npy: 提取的特征数据。
labels.npy: 对应的标签数据。
ML.pkl: 训练好的模型文件。
virus_detection_results.txt: 包含程序运行时间、模型准确率和性能报告。`

##日志记录
程序运行过程中会生成日志信息，记录在控制台中，包括特征提取、模型训练和评估的过程。

##示例
###配置文件示例
json
`{
    "virus_samples_dir": "E:\\样本库\\待拉黑",
    "benign_samples_dir": "E:\\样本库\\待加入白名单",
    "features_path": "features.npy",
    "labels_path": "labels.npy",
    "model_path": "ML.pkl"
}`

###运行结果示例`
2023-10-10 12:34:56,789 - INFO - 模型 ML.pkl 加载成功
2023-10-10 12:34:56,789 - INFO - 发现已存在的特征和标签文件，正在加载...
2023-10-10 12:34:56,789 - INFO - 正在训练模型...
2023-10-10 12:34:56,789 - INFO - 
训练 RandomForest 模型...
Fitting 3 folds for each of 12 candidates, totalling 36 fits
...
2023-10-10 12:34:56,789 - INFO - 最佳模型: RandomForestClassifier
2023-10-10 12:34:56,789 - INFO - 最佳准确率: 0.95
2023-10-10 12:34:56,789 - INFO - 程序运行完毕，总耗时：123.45 秒
2023-10-10 12:34:56,789 - INFO - 最终模型准确率：0.95
`

##贡献
欢迎贡献代码、报告问题或提出建议。
