import os
import time
import numpy as np
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, 
                            roc_curve, PrecisionRecallDisplay, 
                            average_precision_score, confusion_matrix,
                            ConfusionMatrixDisplay)
import psutil
import joblib
import pefile
import math
from imblearn.over_sampling import ADASYN
from concurrent.futures import ProcessPoolExecutor
import logging
import argparse
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                             HistGradientBoostingClassifier)
import io
from PIL import Image
import mmap
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                      VarianceThreshold)
from func_timeout import func_timeout, FunctionTimedOut  
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report, accuracy_score
from collections import Counter
import joblib
import logging
import matplotlib.pyplot as plt
import torch
import pandas as pd
plt.switch_backend('agg')
os.environ['LIGHTGBM_GPU_USE_DOUBLE'] = 'false' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'     
cpu_cores = os.cpu_count()
max_workers = max(1, int(cpu_cores * 0.8))

# 配置日志
log_file = "train_virus_detector.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def calculate_entropy(data):
    """计算字节序列的香农熵"""
    counter = Counter(data)
    length = len(data)
    return -sum((count/length)*math.log2(count/length) for count in counter.values())

def string_to_fixed_length_bytes(s, max_length=100):
    if not s:
        return [0] * max_length
    bytes_ = s.encode('utf-8')[:max_length].ljust(max_length, b'\x00')
    return list(bytes_)

def extract_combined_features(file_path):
    """PE文件特征提取主函数"""
    try:
        return func_timeout(15, _extract_combined_features, args=(file_path,))
    except FunctionTimedOut:
        logging.warning(f"处理文件 {file_path} 超时")
        return None

def _extract_combined_features(file_path):
    try:
        with open(file_path, "rb") as f:
            mmapped_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            file_data = np.frombuffer(mmapped_data, dtype=np.uint8)

        if len(file_data) < 64:
            logging.warning(f"文件 {file_path} 太小，不是有效的 PE 文件")
            return None

        dos_header = file_data[:2].tobytes()
        if dos_header != b'MZ':
            logging.warning(f"文件 {file_path} 缺少 DOS 头签名 (MZ)")
            return None

        pe_header_offset = int.from_bytes(file_data[0x3c:0x40], byteorder='little')
        if pe_header_offset + 4 > len(file_data):
            logging.warning(f"文件 {file_path} 的 PE 头偏移无效")
            return None

        pe_signature = file_data[pe_header_offset:pe_header_offset+4].tobytes()
        if pe_signature != b'PE\x00\x00':
            logging.warning(f"文件 {file_path} 缺少 PE 头签名 (PE\x00\x00)")
            return None

        entropy = calculate_entropy(file_data)

        pe_features = []

        try:
            pe = pefile.PE(file_path)
            pe_features = extract_pe_features(pe)
            file_description = get_pe_string(pe, 'FileDescription')
            file_version = get_pe_string(pe, 'FileVersion')
            product_name = get_pe_string(pe, 'ProductName')
            product_version = get_pe_string(pe, 'ProductVersion')
            legal_copyright = get_pe_string(pe, 'LegalCopyright')

            text_section = next((section for section in pe.sections if section.Name.rstrip(b'\x00').decode('utf-8') == '.text'), None)
            text_section_content = text_section.get_data() if text_section else b''
            text_section_entropy = calculate_entropy(text_section_content) if text_section else 0

            text_section_renamed = 0
            if text_section and text_section.Name.rstrip(b'\x00').decode('utf-8') != '.text':
                text_section_renamed = 1

            data_section = next((section for section in pe.sections if section.Name.rstrip(b'\x00').decode('utf-8') == '.data'), None)
            data_section_size = data_section.Misc_VirtualSize if data_section else 0
            icon_data = []
            icon_color_histograms = []
            icon_dimensions = []
            if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
                for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                    if resource_type.name == pefile.RESOURCE_TYPE['RT_GROUP_ICON']:
                        for resource_id in resource_type.directory.entries:
                            for resource_lang in resource_id.directory.entries:
                                data_rva = resource_lang.data.struct.OffsetToData
                                size = resource_lang.data.struct.Size
                                icon_data.extend(pe.get_memory_mapped_image()[data_rva:data_rva + size])

                                icon_bytes = pe.get_memory_mapped_image()[data_rva:data_rva + size]
                                icon_stream = io.BytesIO(icon_bytes)
                                try:
                                    icon_image = Image.open(icon_stream)
                                    icon_dimensions.append((icon_image.width, icon_image.height))
                                    icon_color_histograms.append(len(icon_image.getcolors()))
                                except Exception as e:
                                    logging.warning(f"无法提取图标特征: {e}")

            file_description_bytes = string_to_fixed_length_bytes(file_description, max_length=50)
            file_version_bytes = string_to_fixed_length_bytes(file_version, max_length=50)
            product_name_bytes = string_to_fixed_length_bytes(product_name, max_length=50)
            product_version_bytes = string_to_fixed_length_bytes(product_version, max_length=50)
            legal_copyright_bytes = string_to_fixed_length_bytes(legal_copyright, max_length=50)

            import_table_info = []
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    import_library_name = entry.dll.decode('utf-8')
                    import_functions = [imp.name.decode('utf-8') for imp in entry.imports if imp.name]
                    import_table_info.append((import_library_name, import_functions))

            export_table_info = []
            if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                    export_function_name = exp.name.decode('utf-8') if exp.name else f"Ordinal_{exp.ordinal}"
                    export_table_info.append((export_function_name, exp.ordinal))

            import_table_features = ','.join([f"{lib}_{func}" for lib, funcs in import_table_info for func in funcs])
            export_table_features = ','.join([f"{name}_{ordinal}" for name, ordinal in export_table_info])

            selected_features = [
                entropy,
                text_section_entropy,
                data_section_size,
                *pe_features,
                *file_description_bytes, *file_version_bytes, *product_name_bytes, *product_version_bytes, *legal_copyright_bytes,
                text_section_renamed
            ]

            for color_count in icon_color_histograms:
                selected_features.append(color_count)
            for width, height in icon_dimensions:
                selected_features.append(width)
                selected_features.append(height)

            expected_length = 1000
            if len(selected_features) < expected_length:
                selected_features.extend([0] * (expected_length - len(selected_features)))
            elif len(selected_features) > expected_length:
                selected_features = selected_features[:expected_length]

            return selected_features
        except pefile.PEFormatError:
            logging.warning(f"文件 {file_path} 不是有效的 PE 文件")
            return None
        except Exception as e:
            logging.error(f"解析 PE 文件 {file_path} 时发生错误: {e}")
            return None
    except Exception as e:
        logging.error(f"无法读取文件 {file_path}: {e}")
        return None

def get_pe_string(pe, string_name):
    if hasattr(pe, 'VS_FIXEDFILEINFO'):
        for entry in pe.FileInfo:
            if hasattr(entry, 'StringTable'):
                for st in entry.StringTable:
                    for key, value in st.entries.items():
                        if key.decode('utf-8') == string_name:
                            return value.decode('utf-8')
            elif hasattr(entry, 'Var'):
                for var in entry.Var:
                    for key, value in var.entry.items():
                        if key.decode('utf-8') == string_name:
                            return value.decode('utf-8')
    return None

def extract_pe_features(pe):
    features = []
    features.append(pe.FILE_HEADER.Machine)
    features.append(pe.FILE_HEADER.NumberOfSections)
    features.append(pe.FILE_HEADER.TimeDateStamp)
    features.append(pe.FILE_HEADER.PointerToSymbolTable)
    features.append(pe.FILE_HEADER.NumberOfSymbols)
    features.append(pe.FILE_HEADER.SizeOfOptionalHeader)
    features.append(pe.FILE_HEADER.Characteristics)
    features.append(pe.OPTIONAL_HEADER.Magic)
    features.append(pe.OPTIONAL_HEADER.MajorLinkerVersion)
    features.append(pe.OPTIONAL_HEADER.MinorLinkerVersion)
    features.append(pe.OPTIONAL_HEADER.SizeOfCode)
    features.append(pe.OPTIONAL_HEADER.SizeOfInitializedData)
    features.append(pe.OPTIONAL_HEADER.SizeOfUninitializedData)
    features.append(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
    features.append(pe.OPTIONAL_HEADER.BaseOfCode)
    features.append(pe.OPTIONAL_HEADER.ImageBase)
    features.append(pe.OPTIONAL_HEADER.SectionAlignment)
    features.append(pe.OPTIONAL_HEADER.FileAlignment)
    features.append(pe.OPTIONAL_HEADER.MajorOperatingSystemVersion)
    features.append(pe.OPTIONAL_HEADER.MinorOperatingSystemVersion)
    features.append(pe.OPTIONAL_HEADER.MajorImageVersion)
    features.append(pe.OPTIONAL_HEADER.MinorImageVersion)
    features.append(pe.OPTIONAL_HEADER.MajorSubsystemVersion)
    features.append(pe.OPTIONAL_HEADER.MinorSubsystemVersion)
    features.append(pe.OPTIONAL_HEADER.SizeOfImage)
    features.append(pe.OPTIONAL_HEADER.SizeOfHeaders)
    features.append(pe.OPTIONAL_HEADER.CheckSum)
    features.append(pe.OPTIONAL_HEADER.Subsystem)
    features.append(pe.OPTIONAL_HEADER.DllCharacteristics)
    features.append(pe.OPTIONAL_HEADER.SizeOfStackReserve)
    features.append(pe.OPTIONAL_HEADER.SizeOfStackCommit)
    features.append(pe.OPTIONAL_HEADER.SizeOfHeapReserve)
    features.append(pe.OPTIONAL_HEADER.LoaderFlags)
    features.append(pe.OPTIONAL_HEADER.NumberOfRvaAndSizes)
    return features

def list_files(directory):
    file_paths = []
    try:
        file_count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
                    file_count += 1
                    if file_count % 100 == 0:
                        logging.info(f"已枚举 {file_count} 个文件...")
        logging.info(f"目录 {directory} 中共找到 {file_count} 个文件")
    except Exception as e:
        logging.error(f"枚举文件时发生错误: {e}")
    return file_paths

def calculate_dynamic_batch_size(total_files):
    """动态计算批处理大小"""
    try:
        base_size = 64
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        
        if total_files == 0:
            return base_size
            
        if available_gb > 8:
            return min(256, total_files)
        elif available_gb > 4:
            return min(128, total_files)
        else:
            return min(64, total_files)
    except Exception as e:
        logging.warning(f"动态计算批处理大小失败: {e}, 使用默认值64")
        return 64


def create_dataset(virus_dir, benign_dir, batch_size=None, max_workers=max_workers):
    """
    创建训练数据集
    特征处理：
    - 多进程并行处理
    - 自动填充特征到1000维
    - 超时处理（15秒/文件）
    - 错误文件自动跳过
    """
    if not os.path.exists(virus_dir):
        logging.error(f"病毒样本目录 {virus_dir} 不存在")
        return [], []
    if not os.path.exists(benign_dir):
        logging.error(f"良性样本目录 {benign_dir} 不存在")
        return [], []

    # 获取文件列表
    virus_files = list_files(virus_dir)
    benign_files = list_files(benign_dir)

    if not virus_files:
        logging.warning(f"病毒样本目录 {virus_dir} 中没有文件")
    if not benign_files:
        logging.warning(f"良性样本目录 {benign_dir} 中没有文件")

    all_files = [(file_path, 1) for file_path in virus_files] + [(file_path, 0) for file_path in benign_files]

    # 动态计算batch_size
    if batch_size is None:
        batch_size = calculate_dynamic_batch_size(len(all_files))
        logging.info(f"动态计算批处理大小: {batch_size}")
    
    if batch_size <= 0:
        raise ValueError(f"无效的批处理大小: {batch_size}")

    total_batches = (len(all_files) + batch_size - 1) // batch_size
    logging.info(f"开始生成数据集，总文件数: {len(all_files)}，批处理大小: {batch_size}, 总批次数: {total_batches}")

    virus_features_all = []
    benign_features_all = []

    for batch_idx in tqdm(range(total_batches), desc="处理批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_files))
        batch_files = all_files[start_idx:end_idx]

        with ProcessPoolExecutor(max_workers=max_workers) as executor: 
            futures = [executor.submit(process_file_wrapper, (file_path, label)) for (file_path, label) in batch_files]
            results = []
            for future in tqdm(futures, desc="处理批次文件"):
                try:
                    results.append(future.result(timeout=15))
                except TimeoutError:
                    logging.error("处理文件超时")
                    results.append((None, None))

        skipped_files = [file_path for (file_path, label), result in zip(batch_files, results) if result[0] is None]
        if skipped_files:
            logging.warning(f"跳过文件: {skipped_files}")

        virus_features_batch = [feature for feature, label in results if feature is not None and label == 1]
        benign_features_batch = [feature for feature, label in results if feature is not None and label == 0]

        virus_features_all.extend(virus_features_batch)
        benign_features_all.extend(benign_features_batch)

        logging.info(f"批次 {batch_idx + 1}/{total_batches} 处理完成，病毒样本数: {len(virus_features_batch)}，良性样本数: {len(benign_features_batch)}")

    logging.info(f"数据集生成完成，总病毒样本数: {len(virus_features_all)}，总良性样本数: {len(benign_features_all)}")

    expected_length = 1000
    virus_features_all = [feature + [0] * (expected_length - len(feature)) if len(feature) < expected_length else feature for feature in virus_features_all]
    benign_features_all = [feature + [0] * (expected_length - len(feature)) if len(feature) < expected_length else feature for feature in benign_features_all]

    def validate_features(feature_list):
        return [f + [0]*(1000-len(f)) if len(f)<1000 else f[:1000] 
                for f in feature_list]

    virus_features_all = validate_features(virus_features_all)
    benign_features_all = validate_features(benign_features_all)
    
    return virus_features_all, benign_features_all


def create_advanced_feature_pipeline():
    """创建增强型特征工程流水线"""
    return Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.05*(1-0.05))),
        ('mutual_info', SelectKBest(score_func=mutual_info_classif, k=500)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)), 
    ])

def create_optimized_model_pipeline():
    """创建优化后的模型流水线"""
    # LightGBM参数
    lgbm_params = {
        'num_leaves': 127,
        'max_depth': -1,
        'learning_rate': 0.03,
        'n_estimators': 1500,
        'reg_alpha': 0.5,
        'reg_lambda': 0.7,
        'min_child_samples': 40,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'class_weight': {0:1, 1:5},
        'objective': 'binary',
        'metric': 'aucpr',
        'boosting_type': 'goss',
        'device': device,
        'early_stopping_round': None,
        'verbosity': -1
    }

    # XGBoost参数
    xgb_params = {
        'max_depth': 7,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'device': device,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'gamma': 0.2
    }

    # SVM参数
    svm_params = {
        'dual': False,
        'C': 0.5,
        'class_weight': 'balanced',
        'max_iter': 1000
    }

    # 最终评估器
    final_estimator = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=5,
        categorical_features=None
    )

    # 分离特征工程和模型训练
    feature_pipeline = create_advanced_feature_pipeline()
    
    return Pipeline([
        ('feature_engineering', feature_pipeline),
        ('classifier', StackingClassifier(  
            estimators=[
                ('lgbm', LGBMClassifier(**lgbm_params)),
                ('xgb', XGBClassifier(**xgb_params)),
                ('svm', LinearSVC(**svm_params))
            ],
            final_estimator=final_estimator,
            passthrough=False, 
            stack_method='auto',
            n_jobs=4
        ))
    ])

def enhanced_evaluation(y_true, y_prob, y_pred):
    """增强型模型评估"""
    results = {}
    
    # PR曲线
    pr_display = PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    results['pr_curve'] = pr_display
    
    # 关键指标
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    results['tpr_at_1fpr'] = tpr[fpr <= 0.01][-1] if any(fpr <= 0.01) else 0.0
    results['average_precision'] = average_precision_score(y_true, y_prob)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(cm)
    results['confusion_matrix'] = cm_display
    
    return results

def process_file_wrapper_with_vectorizer(args):
    return process_file_wrapper(args)

def process_file(file_path, label):
    try:
        feature = extract_combined_features(file_path)
        return (feature, label) if feature is not None else (None, None)
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时发生错误: {e}")
        return (None, None)

def process_file_wrapper(args):
    try:
        return process_file(*args)
    except Exception as e:
        logging.error(f"处理文件 {args[0]} 时发生错误: {e}")
        return (None, None)

def merge_features_and_labels(virus_features, benign_features):
    """合并特征和标签"""
    virus_features = virus_features if virus_features else []
    benign_features = benign_features if benign_features else []
    
    features = virus_features + benign_features
    labels = [1] * len(virus_features) + [0] * len(benign_features)
    
    if len(features) == 0:
        return np.empty((0, 1000)), np.empty(0)
    
    features_array = np.array(features)
    if features_array.shape[1] != 1000:
        features_array = np.pad(features_array, 
                              ((0,0), (0,1000 - features_array.shape[1])), 
                              mode='constant')
    
    logging.info(f"特征和标签合并完成，总样本数: {len(features)}")
    return features_array, np.array(labels)


def save_features_and_labels(features, labels, features_path, labels_path):
    np.save(features_path, features)
    np.save(labels_path, labels)
    logging.info(f"特征和标签已保存到 {features_path} 和 {labels_path}")

def load_features_and_labels(features_path, labels_path):
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        logging.error(f"特征或标签文件不存在: {features_path}, {labels_path}")
        return None, None
    features = np.load(features_path)
    labels = np.load(labels_path)
    logging.info(f"特征和标签已从 {features_path} 和 {labels_path} 加载")
    return features, labels

def preprocess_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

def compress_features(features, method='PCA', n_components=50):
    if method == 'PCA':
        from sklearn.decomposition import PCA
        compressor = PCA(n_components=0.95)
        compressed_features = compressor.fit_transform(features)
        logging.info(f"特征压缩完成，压缩方法: {method}, 保留的主成分数量: {compressed_features.shape[1]}")
    elif method == 'AutoEncoder':
        from sklearn.neural_network import MLPRegressor
        compressor = MLPRegressor(hidden_layer_sizes=(n_components,), activation='relu', solver='adam', max_iter=500)
        compressed_features = compressor.fit_transform(features)
    else:
        raise ValueError("Unsupported compression method")

    return compressed_features

def augment_data(features, labels):
    logging.info("开始数据增强...")
    try:
        ada = ADASYN(random_state=42)
        features_resampled, labels_resampled = ada.fit_resample(features, labels)
        logging.info("数据增强完成，增强后的样本数: {}".format(len(labels_resampled)))
        return features_resampled, labels_resampled
    except Exception as e:
        logging.error(f"数据增强时发生错误: {e}")
        return features, labels


# 保存特征名称
def save_feature_names(features, path):
    feature_names = [f'feature_{i}' for i in range(features.shape[1])]
    with open(path, 'w') as f:
        json.dump(feature_names, f)
    logging.info(f"特征名称已保存到 {path}")

# 加载特征名称
def load_feature_names(path):
    with open(path, 'r') as f:
        feature_names = json.load(f)
    logging.info(f"特征名称已从 {path} 加载")
    return feature_names


def incremental_training(existing_features, existing_labels, new_features, new_labels):
    """增强型增量训练（修复版本）"""
    # 转换为numpy数组
    existing_features = np.array(existing_features) if existing_features is not None else np.empty((0, 1000))
    existing_labels = np.array(existing_labels) if existing_labels is not None else np.empty(0)
    new_features = np.array(new_features)
    new_labels = np.array(new_labels)
    
    # 维度验证和调整
    def adjust_dimensions(features):
        if features.shape[1] < 1000:
            return np.pad(features, ((0,0), (0,1000 - features.shape[1])), mode='constant')
        elif features.shape[1] > 1000:
            return features[:, :1000]
        return features
    
    # 调整所有特征的维度
    if existing_features.size > 0:
        existing_features = adjust_dimensions(existing_features)
    if new_features.size > 0:
        new_features = adjust_dimensions(new_features)
    
    # 合并数据集
    if existing_features.size == 0:
        features = new_features
        labels = new_labels
    elif new_features.size == 0:
        features = existing_features
        labels = existing_labels
    else:
        features = np.vstack([existing_features, new_features])
        labels = np.concatenate([existing_labels, new_labels])
    
    # 动态调整样本权重
    if len(labels) > 0:
        malware_ratio = np.mean(labels)
        sample_weights = np.where(labels == 1, 
                                1/(malware_ratio + 1e-5), 
                                1/(1 - malware_ratio + 1e-5))
    else:
        sample_weights = np.empty(0)
    
    assert features.shape[1] == 1000, f"特征维度错误，应为1000维，实际得到{features.shape[1]}维"
    assert len(features) == len(labels), "特征和标签数量不匹配"
    return features, labels



def train_model(features, labels):
    """优化后的模型训练流程"""
    logging.info(f"当前内存使用：{psutil.virtual_memory().percent}%")
    logging.info(f"数据集大小: {len(features)}")
    logging.info(f"类别分布: {Counter(labels)}")
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # 贝叶斯优化参数空间
    param_space = {
        'classifier__lgbm__num_leaves': Integer(50, 200),  
        'classifier__xgb__gamma': Real(0, 0.5),
        'feature_engineering__pca__n_components': Real(0.7, 0.95),
        'feature_engineering__mutual_info__k': Integer(300, 800)
    }

    # 创建优化器
    optimizer = BayesSearchCV(
        estimator=create_optimized_model_pipeline(),
        search_spaces=param_space,
        n_iter=100,
        cv=5,
        n_jobs=6,
        verbose=2,
        scoring='average_precision',
        random_state=42
    )
    
    # 训练模型
    optimizer.fit(X_train, y_train)
    best_model = optimizer.best_estimator_
    
    # 加载特征名称
    feature_names = load_feature_names('feature_names.json')
    
    # 将测试数据转换为 DataFrame
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # 模型评估
    y_pred = model.predict(X_new_df)
    y_pred = (y_prob >= 0.5).astype(int)
    eval_results = enhanced_evaluation(y_test, y_prob, y_pred)
    
    # 保存关键评估结果
    logging.info(f"平均精度(AP): {eval_results['average_precision']:.3f}")
    logging.info(f"1% FPR下的检测率: {eval_results['tpr_at_1fpr']:.3f}")
    logging.info(f"分类报告:\n{classification_report(y_test, y_pred)}")
    
    # 保存模型组件
    joblib.dump(best_model, 'optimized_model.joblib')
    logging.info("优化后的模型已保存")
    
    return best_model, eval_results

def save_model(model, output_path):
    joblib.dump(model, output_path)
    logging.info(f"模型已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恶意软件检测模型训练")
    parser.add_argument('--config', type=str, required=True, help="配置文件路径")
    parser.add_argument('--mode', type=str, choices=['incremental', 'fine_tune'], default='incremental', help="训练模式：增量训练或微调")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logging.error(f"配置文件 {args.config} 不存在")
        exit(1)

    with open(args.config, 'r') as f:
        try:
            config = json.load(f)
            logging.info(f"配置文件内容: {config}")
        except json.JSONDecodeError as e:
            logging.error(f"配置文件格式错误: {e}")
            exit(1)
    # 检查是否有可用的Nvidia GPU
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("检测到Nvidia GPU，使用GPU进行训练")
    else:
        device = 'cpu'
        logging.info("未检测到Nvidia GPU，使用CPU进行训练")
    
    virus_samples_dir = config.get("virus_samples_dir", "E:\\样本库\\待拉黑")
    benign_samples_dir = config.get("benign_samples_dir", "E:\\样本库\\待加入白名单")
    features_path = config.get("features_path", "features.npy")
    labels_path = config.get("labels_path", "labels.npy")
    model_output_path = config.get("model_output_path", "model.joblib")

    start_time = time.time()
    if args.mode == 'incremental':
        logging.info("增量训练模式")
        existing_features, existing_labels = load_features_and_labels(features_path, labels_path)
        if existing_features is None:
            existing_features = np.empty((0, 1000))
            existing_labels = np.empty(0)
        else:
            existing_features = np.array(existing_features)
            existing_labels = np.array(existing_labels)

    # 生成新数据集
        virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir)
        new_features, new_labels = merge_features_and_labels(virus_features, benign_features)
        new_features = np.array(new_features) if len(new_features) > 0 else np.empty((0, 1000))
        new_labels = np.array(new_labels) if len(new_labels) > 0 else np.empty(0)
    
    # 执行增强型增量训练
        features, labels = incremental_training(
            existing_features, existing_labels, 
            new_features, new_labels
        )
    
        if features.size > 0 and labels.size > 0:
            save_features_and_labels(features, labels, features_path, labels_path)
            save_feature_names(features, 'feature_names.json')
            logging.info(f"合并后数据集尺寸: 特征{features.shape}, 标签{labels.shape}")


        # 数据增强
        features, labels = augment_data(features, labels)
        
        model, eval_results = train_model(features, labels)
        
        joblib.dump(model, model_output_path)
        logging.info(f"模型已保存至 {model_output_path}")
        
        pr_fig = eval_results['pr_curve'].figure_
        pr_fig.savefig('precision_recall_curve.png')

        cm_display = eval_results['confusion_matrix']
        cm_display.plot()
        cm_display.figure_.savefig('confusion_matrix.png')  
        
    else:
        logging.info("微调模式")
        if os.path.exists(features_path) and os.path.exists(labels_path):
            logging.info(f"特征和标签文件已存在: {features_path}, {labels_path}")
            existing_features, existing_labels = load_features_and_labels(features_path, labels_path)
            if existing_features is None or existing_labels is None:
                logging.error("无法加载特征和标签，重新生成特征和标签...")
                virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir)
                features, labels = merge_features_and_labels(virus_features, benign_features)
            else:
                virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir)
                new_features, new_labels = merge_features_and_labels(virus_features, benign_features)

                if len(new_features) == 0 and len(new_labels) == 0:
                    logging.info("新生成的特征和标签数量为0，跳过合并数据集，直接使用现有数据集开始训练")
                    features = existing_features
                    labels = existing_labels
                else:
                    features = np.vstack((existing_features, new_features))
                    labels = np.concatenate((existing_labels, new_labels))
                    logging.info(f"新生成的特征和标签已合并到现有数据集中，总样本数: {len(features)}")

                save_features_and_labels(features, labels, features_path, labels_path)
                save_feature_names(features, 'feature_names.json')
        else:
            virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir)
            features, labels = merge_features_and_labels(virus_features, benign_features)

            save_features_and_labels(features, labels, features_path, labels_path)
            save_feature_names(features, 'feature_names.json')

        features, labels = augment_data(features, labels)
        initial_model = joblib.load(model_output_path)
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        model, accuracy, y_test, y_pred = train_model(features, labels, initial_model)
        save_model(model, model_output_path)

    end_time = time.time()
    logging.info(f"训练完成，总耗时: {end_time - start_time:.2f} 秒")
