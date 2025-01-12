import os
import time
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import joblib
import pefile
import math
from imblearn.over_sampling import ADASYN
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse
import json
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 计算数据的熵值
def calculate_entropy(data):
    if not data:
        return 0
    counter = Counter(data)
    length = len(data)
    entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
    return entropy

# 提取PE文件的特征
def extract_pe_features(pe):
    try:
        dos_header = pe.DOS_HEADER
        file_header = pe.FILE_HEADER
        optional_header = pe.OPTIONAL_HEADER

        features = [
            dos_header.e_magic,
            dos_header.e_lfanew,
            file_header.Machine,
            file_header.NumberOfSections,
            optional_header.AddressOfEntryPoint,
            optional_header.ImageBase,
            optional_header.SectionAlignment,
            optional_header.FileAlignment,
            optional_header.SizeOfImage,
            optional_header.SizeOfHeaders,
            optional_header.CheckSum,
            optional_header.Subsystem,
            optional_header.SizeOfStackReserve,
            optional_header.SizeOfStackCommit,
            optional_header.SizeOfHeapReserve,
            optional_header.SizeOfHeapCommit,
            optional_header.NumberOfRvaAndSizes
        ]

        # 提取 .Text 段的二进制数据并计算其熵值
        text_section = next((section for section in pe.sections if b'.text' in section.Name), None)
        text_entropy = calculate_entropy(text_section.get_data()) if text_section else 0
        features.append(text_entropy)

        # 提取 .data 段的大小
        data_section_size = next((section.Misc_VirtualSize for section in pe.sections if b'.data' in section.Name), 0)
        features.append(data_section_size)

        # 提取导入表中的前32个函数名的前50字节
        imported_functions = set()
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        imported_functions.add(imp.name.decode('utf-8', errors='ignore'))
        
        sorted_imported_functions = sorted(imported_functions)[:32]
        function_names_feature = [func.encode('utf-8')[:50] for func in sorted_imported_functions]
        function_names_feature += [b'\x00'] * (32 - len(function_names_feature))
        function_names_feature = [byte for func_name in function_names_feature for byte in func_name.ljust(50, b'\x00')]
        features.extend(function_names_feature)

        # 提取文件说明和版权信息
        description = ''
        copyright_info = ''
        if hasattr(pe, 'FileInfo'):
            for entry in pe.FileInfo:
                if hasattr(entry, 'StringTable'):
                    for st in entry.StringTable:
                        for key, value in st.entries.items():
                            if key.lower() == 'filedescription':
                                description = value.decode('utf-8', errors='ignore')
                            elif key.lower() == 'legalcopyright':
                                copyright_info = value.decode('utf-8', errors='ignore')
        
        description_bytes = description.encode('utf-8')[:128] + b'\x00' * (128 - min(len(description), 128))
        copyright_info_bytes = copyright_info.encode('utf-8')[:128] + b'\x00' * (128 - min(len(copyright_info), 128))
        features.extend(description_bytes)
        features.extend(copyright_info_bytes)

        return features
    except Exception as e:
        logging.error(f"无法提取 PE 特征: {e}")
        return None

# 提取文件的特征
def extract_features(file_path):
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()

        byte_distribution = [0] * 256
        for byte in file_data:
            byte_distribution[byte] += 1
        byte_distribution = [x / len(file_data) for x in byte_distribution]

        entropy = calculate_entropy(file_data)

        first_128_bytes = file_data[:128].ljust(128, b'\x00')
        last_128_bytes = file_data[-128:].rjust(128, b'\x00')

        pe_features = []
        try:
            pe = pefile.PE(file_path)
            pe_features = extract_pe_features(pe)
        except pefile.PEFormatError:
            logging.warning(f"文件 {file_path} 不是有效的 PE 文件")
            return None

        selected_features = [entropy] + byte_distribution + list(first_128_bytes) + list(last_128_bytes) + pe_features

        return selected_features
    except Exception as e:
        logging.error(f"无法读取文件 {file_path}: {e}")
        return None

# 处理单个文件，返回其特征和标签
def process_file(file_path, label):
    feature = extract_features(file_path)
    return (feature, label) if feature is not None else (None, None)

# 创建数据集并分批处理
def create_dataset(virus_dir, benign_dir, batch_size=32, max_workers=None):
    virus_files = [os.path.join(virus_dir, file_name) for file_name in os.listdir(virus_dir)]
    benign_files = [os.path.join(benign_dir, file_name) for file_name in os.listdir(benign_dir)]

    all_files = [(file_path, 1) for file_path in virus_files] + [(file_path, 0) for file_path in benign_files]

    total_batches = (len(all_files) + batch_size - 1) // batch_size
    virus_features_all = []
    benign_features_all = []

    for batch_idx in tqdm(range(total_batches), desc="处理批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_files))
        batch_files = all_files[start_idx:end_idx]

        with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            results = list(executor.map(lambda args: process_file(*args), batch_files))

        virus_features_batch = [feature for feature, label in results if feature is not None and label == 1]
        benign_features_batch = [feature for feature, label in results if feature is not None and label == 0]

        virus_features_all.extend(virus_features_batch)
        benign_features_all.extend(benign_features_batch)

    return virus_features_all, benign_features_all

# 合并所有批次的特征和标签
def merge_features_and_labels(virus_features, benign_features):
    features = virus_features + benign_features
    labels = [1] * len(virus_features) + [0] * len(benign_features)
    return np.array(features), np.array(labels)

# 保存特征和标签
def save_features_and_labels(features, labels, features_path, labels_path):
    np.save(features_path, features)
    np.save(labels_path, labels)
    logging.info(f"特征和标签已保存到 {features_path} 和 {labels_path}")

# 加载特征和标签
def load_features_and_labels(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    logging.info(f"特征和标签已从 {features_path} 和 {labels_path} 加载")
    return features, labels

# 数据增强
def augment_data(X, y, n_samples=1000):
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

# 训练模型
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

    param_grids = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs']
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
        }
    }

    best_model = None
    best_accuracy = 0
    best_y_pred = None

    for model_name, config in param_grids.items():
        logging.info(f"\n训练 {model_name} 模型...")
        grid_search = GridSearchCV(estimator=config['model'], param_grid=config['param_grid'], cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_resampled, y_train_resampled)

        best_estimator = grid_search.best_estimator_
        y_prob = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, 'predict_proba') else best_estimator.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_prob >= optimal_threshold).astype(int)

        logging.info(f"{model_name} 最佳参数: {grid_search.best_params_}")
        logging.info(f"{model_name} 最优阈值: {optimal_threshold}")
        logging.info(f"{model_name} 模型性能报告：\n{classification_report(y_test, y_pred)}")
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{model_name} 准确率：{accuracy:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_estimator
            best_y_pred = y_pred

    logging.info(f"\n最佳模型: {type(best_model).__name__}")
    logging.info(f"最佳准确率: {best_accuracy:.2f}")

    return best_model, best_accuracy, y_test, best_y_pred

# 保存模型
def save_model(model, output_path):
    joblib.dump(model, output_path)
    logging.info(f"模型已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恶意软件检测模型训练")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    virus_samples_dir = config.get("virus_samples_dir", "E:\\样本库\\待拉黑")
    benign_samples_dir = config.get("benign_samples_dir", "E:\\样本库\\待加入白名单")
    features_path = config.get("features_path", "features.npy")
    labels_path = config.get("labels_path", "labels.npy")

    start_time = time.time()

    if os.path.exists(features_path) and os.path.exists(labels_path):
        logging.info("发现已存在的特征和标签文件，正在加载...")
        features, labels = load_features_and_labels(features_path, labels_path)
    else:
        logging.info("未找到特征和标签文件，正在创建数据集...")
        virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir, batch_size=256)
        
        if not virus_features or not benign_features:
            logging.error("数据集为空，请检查样本路径或样本内容。")
        else:
            features, labels = merge_features_and_labels(virus_features, benign_features)
            save_features_and_labels(features, labels, features_path, labels_path)

    if features.size != 0 and labels.size != 0:
        logging.info("正在训练模型...")
        model, accuracy, y_test, y_pred = train_model(features, labels)

        model_path = config.get("model_path", "ML.pkl")
        save_model(model, model_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"程序运行完毕，总耗时：{elapsed_time:.2f} 秒")
        logging.info(f"最终模型准确率：{accuracy:.2f}")

        with open("virus_detection_results.txt", "w", encoding="utf-8") as f:
            f.write(f"程序运行完毕，总耗时：{elapsed_time:.2f} 秒\n")
            f.write(f"最终模型准确率：{accuracy:.2f}\n")
            f.write("模型性能报告：\n")
            f.write(classification_report(y_test, y_pred))