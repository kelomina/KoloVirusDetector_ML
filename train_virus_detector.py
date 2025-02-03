import os
import time
import numpy as np
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve
import joblib
import pefile
import math
from imblearn.over_sampling import ADASYN, SMOTE
from concurrent.futures import ProcessPoolExecutor
import logging
import argparse
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import io
from PIL import Image
from scipy.stats import anderson_ksamp
import mmap
from sklearn.feature_selection import SelectKBest, f_classif
from func_timeout import func_timeout, FunctionTimedOut  # 导入 func_timeout 和 FunctionTimedOut
from sklearn.decomposition import PCA  # 添加 PCA 导入

# 配置日志记录
log_file = "train_virus_detector.log"
logging.basicConfig(
    level=logging.DEBUG,  # 强制 DEBUG 级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # 统一日志时间格式
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 计算数据的熵
def calculate_entropy(data):
    counter = Counter(data)
    length = len(data)
    entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
    return entropy

# 字符串特征定长处理
def string_to_fixed_length_bytes(s, max_length=100):
    if not s:
        return [0] * max_length
    bytes_ = s.encode('utf-8')[:max_length].ljust(max_length, b'\x00')
    return list(bytes_)

# 提取文件特征
def extract_combined_features(file_path):
    try:
        return func_timeout(30, _extract_combined_features, args=(file_path,))  # 30秒超时
    except FunctionTimedOut:
        logging.error(f"处理文件 {file_path} 超时")
        return None

def _extract_combined_features(file_path):
    try:
        with open(file_path, "rb") as f:
            mmapped_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            file_data = np.frombuffer(mmapped_data, dtype=np.uint8)

        # 检查文件大小
        if len(file_data) < 64:
            logging.warning(f"文件 {file_path} 太小，不是有效的 PE 文件")
            return None

        # 检查 DOS 头（转换为 bytes 比较）
        dos_header = file_data[:2].tobytes()
        if dos_header != b'MZ':
            logging.warning(f"文件 {file_path} 缺少 DOS 头签名 (MZ)")
            return None

        # 检查 PE 头位置有效性
        pe_header_offset = int.from_bytes(file_data[0x3c:0x40], byteorder='little')
        if pe_header_offset + 4 > len(file_data):
            logging.warning(f"文件 {file_path} 的 PE 头偏移无效")
            return None

        # 检查 PE 签名（转换为 bytes 比较）
        pe_signature = file_data[pe_header_offset:pe_header_offset+4].tobytes()
        if pe_signature != b'PE\x00\x00':
            logging.warning(f"文件 {file_path} 缺少 PE 头签名 (PE\x00\x00)")
            return None

        # 计算文件熵
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

            # 检查数字签名和目录签名
            has_digital_signature = 0
            has_directory_signature = 0
            if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY'):
                has_directory_signature = 1
                if pe.DIRECTORY_ENTRY_SECURITY:
                    has_digital_signature = 1
            # 检查数字签名有效性
            digital_signature_valid = 0
            if has_digital_signature:
                try:
                    if pe.verify_signature():
                        digital_signature_valid = 1
                except Exception as e:
                    logging.warning(f"文件 {file_path} 数字签名验证失败: {e}")
            # 提取 .text 段内容
            text_section = next((section for section in pe.sections if section.Name.rstrip(b'\x00').decode('utf-8') == '.text'), None)
            text_section_content = text_section.get_data() if text_section else b''
            text_section_entropy = calculate_entropy(text_section_content) if text_section else 0

            # 检查 .text 段是否被重命名
            text_section_renamed = 0
            if text_section and text_section.Name.rstrip(b'\x00').decode('utf-8') != '.text':
                text_section_renamed = 1

            # 提取 .data 段大小
            data_section = next((section for section in pe.sections if section.Name.rstrip(b'\x00').decode('utf-8') == '.data'), None)
            data_section_size = data_section.Misc_VirtualSize if data_section else 0
            # 提取图标数据
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

                                # 提取图标颜色直方图和尺寸
                                icon_bytes = pe.get_memory_mapped_image()[data_rva:data_rva + size]
                                icon_stream = io.BytesIO(icon_bytes)
                                try:
                                    icon_image = Image.open(icon_stream)
                                    icon_dimensions.append((icon_image.width, icon_image.height))
                                    icon_color_histograms.append(len(icon_image.getcolors()))
                                except Exception as e:
                                    logging.warning(f"无法提取图标特征: {e}")

            # 将字符串特征转换为字节码特征
            file_description_bytes = string_to_fixed_length_bytes(file_description, max_length=50)
            file_version_bytes = string_to_fixed_length_bytes(file_version, max_length=50)
            product_name_bytes = string_to_fixed_length_bytes(product_name, max_length=50)
            product_version_bytes = string_to_fixed_length_bytes(product_version, max_length=50)
            legal_copyright_bytes = string_to_fixed_length_bytes(legal_copyright, max_length=50)

            # 提取导入表信息
            import_table_info = []
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    import_library_name = entry.dll.decode('utf-8')
                    import_functions = [imp.name.decode('utf-8') for imp in entry.imports if imp.name]
                    import_table_info.append((import_library_name, import_functions))

            # 提取导出表信息
            export_table_info = []
            if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                    export_function_name = exp.name.decode('utf-8') if exp.name else f"Ordinal_{exp.ordinal}"
                    export_table_info.append((export_function_name, exp.ordinal))

            # 合并导入表和导出表信息为一个字符串
            import_table_features = ','.join([f"{lib}_{func}" for lib, funcs in import_table_info for func in funcs])
            export_table_features = ','.join([f"{name}_{ordinal}" for name, ordinal in export_table_info])

            # 合并特征
            selected_features = [
                entropy,
                text_section_entropy,
                data_section_size,
                *pe_features,
                *file_description_bytes, *file_version_bytes, *product_name_bytes, *product_version_bytes, *legal_copyright_bytes,
                has_digital_signature, has_directory_signature, digital_signature_valid, text_section_renamed
            ]

            # 添加图标颜色直方图和尺寸特征
            for color_count in icon_color_histograms:
                selected_features.append(color_count)
            for width, height in icon_dimensions:
                selected_features.append(width)
                selected_features.append(height)

            # 确保所有特征向量长度一致
            expected_length = 1000  # 根据实际情况调整
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

# 获取 PE 文件的字符串信息
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

# 提取 PE 文件特征
def extract_pe_features(pe):
    features = []
    # 提取 PE 文件的基本信息
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
    #features.append(pe.OPTIONAL_HEADER.Win32VersionValue)
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

def create_dataset(virus_dir, benign_dir, batch_size=512, max_workers=512):
    if not os.path.exists(virus_dir):
        logging.error(f"病毒样本目录 {virus_dir} 不存在")
        return [], []
    if not os.path.exists(benign_dir):
        logging.error(f"良性样本目录 {benign_dir} 不存在")
        return [], []

    virus_files = list_files(virus_dir)
    benign_files = list_files(benign_dir)

    if not virus_files:
        logging.warning(f"病毒样本目录 {virus_dir} 中没有文件")
    if not benign_files:
        logging.warning(f"良性样本目录 {benign_dir} 中没有文件")

    all_files = [(file_path, 1) for file_path in virus_files] + [(file_path, 0) for file_path in benign_files]

    total_batches = (len(all_files) + batch_size - 1) // batch_size
    virus_features_all = []
    benign_features_all = []

    logging.info(f"开始生成数据集，总文件数: {len(all_files)}，批处理大小: {batch_size}")

    for batch_idx in tqdm(range(total_batches), desc="处理批次"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_files))
        batch_files = all_files[start_idx:end_idx]

        with ProcessPoolExecutor(max_workers=4) as executor:  # 减少并发数
            futures = [executor.submit(process_file_wrapper, (file_path, label)) for (file_path, label) in batch_files]
            results = []
            for future in tqdm(futures, desc="处理批次文件"):
                try:
                    results.append(future.result(timeout=60))  # 增加超时时间到 60 秒
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

    # 补0到预期长度
    expected_length = 1000  # 根据实际情况调整
    virus_features_all = [feature + [0] * (expected_length - len(feature)) if len(feature) < expected_length else feature for feature in virus_features_all]
    benign_features_all = [feature + [0] * (expected_length - len(feature)) if len(feature) < expected_length else feature for feature in benign_features_all]

    return virus_features_all, benign_features_all

# 新增命名函数
def process_file_wrapper_with_vectorizer(args):
    return process_file_wrapper(args)

# 处理单个文件
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

# 合并特征和标签
def merge_features_and_labels(virus_features, benign_features):
    features = virus_features + benign_features
    labels = [1] * len(virus_features) + [0] * len(benign_features)
    logging.info(f"特征和标签合并完成，总样本数: {len(features)}")
    return np.array(features), np.array(labels)

# 保存特征和标签
def save_features_and_labels(features, labels, features_path, labels_path):
    np.save(features_path, features)
    np.save(labels_path, labels)
    logging.info(f"特征和标签已保存到 {features_path} 和 {labels_path}")

# 加载特征和标签
def load_features_and_labels(features_path, labels_path):
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        logging.error(f"特征或标签文件不存在: {features_path}, {labels_path}")
        return None, None
    features = np.load(features_path)
    labels = np.load(labels_path)
    logging.info(f"特征和标签已从 {features_path} 和 {labels_path} 加载")
    return features, labels

# 特征预处理
def preprocess_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

# 特征压缩
def compress_features(features, method='PCA', n_components=50):
    if method == 'PCA':
        from sklearn.decomposition import PCA
        compressor = PCA(n_components=0.95)  # 保留95%的方差
        compressed_features = compressor.fit_transform(features)
        logging.info(f"特征压缩完成，压缩方法: {method}, 保留的主成分数量: {compressed_features.shape[1]}")
    elif method == 'AutoEncoder':
        from sklearn.neural_network import MLPRegressor
        compressor = MLPRegressor(hidden_layer_sizes=(n_components,), activation='relu', solver='adam', max_iter=500)
        compressed_features = compressor.fit_transform(features)
    else:
        raise ValueError("Unsupported compression method")

    return compressed_features

# 定义数据增强函数
def augment_data(features, labels):
    logging.info("开始数据增强...")
    try:
        # 使用 ADASYN 进行数据增强
        ada = ADASYN(random_state=42)
        features_resampled, labels_resampled = ada.fit_resample(features, labels)
        logging.info("数据增强完成，增强后的样本数: {}".format(len(labels_resampled)))
        return features_resampled, labels_resampled
    except Exception as e:
        logging.error(f"数据增强时发生错误: {e}")
        return features, labels

# 训练模型
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

def train_model(features, labels, initial_model=None, feature_importance_threshold=0.01, min_features=10):
    logging.info(f"数据集大小: {len(features)}")
    logging.info(f"类别分布: {Counter(labels)}")

    # 特征预处理
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 特征压缩
    pca = PCA(n_components=0.95)
    features_compressed = pca.fit_transform(features_scaled)

    X_train, X_test, y_train, y_test = train_test_split(features_compressed, labels, test_size=0.2, random_state=42)

    # 使用Pipeline嵌入特征选择步骤
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', StackingClassifier(
            estimators=[('LightGBM', LGBMClassifier(random_state=42, class_weight='balanced', num_leaves=31, min_data_in_leaf=20)),
                        ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', class_weight='balanced')),
                        ('RandomForest', RandomForestClassifier(random_state=42, class_weight='balanced'))],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', class_weight='balanced'),
            stack_method='predict_proba'
        ))
    ])

    param_grid = {
        'feature_selection__k': [50, 100, 'all'],
        'classifier__LightGBM__n_estimators': [100, 200, 500],
        'classifier__LightGBM__max_depth': [5, 7, 9],
        'classifier__LightGBM__reg_alpha': [0, 0.1],
        'classifier__LightGBM__reg_lambda': [0, 0.1],
        'classifier__LightGBM__min_child_samples': [20, 50],
        'classifier__LogisticRegression__C': [0.1, 1, 10],
        'classifier__RandomForest__n_estimators': [50, 100, 200],
        'classifier__RandomForest__max_depth': [3, 5, 7, 9],
        'classifier__final_estimator__C': [0.1, 1, 10]
    }

    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=4, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_estimator = random_search.best_estimator_
    y_prob = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, 'predict_proba') else best_estimator.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    logging.info(f"Stacking模型最佳参数: {random_search.best_params_}")
    logging.info(f"Stacking模型最优阈值: {optimal_threshold}")
    logging.info(f"Stacking模型性能报告：\n{classification_report(y_test, y_pred)}")
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Stacking模型准确率：{accuracy:.2f}")

    # 获取LightGBM特征重要性
    lgbm_model = best_estimator.named_steps['classifier'].named_estimators_['LightGBM']
    feature_importances = lgbm_model.feature_importances_
    feature_names = np.array([f"feature_{i}" for i in range(X_train.shape[1])])

    # 递归地去除不重要的特征
    while len(feature_importances) > min_features and np.any(feature_importances < feature_importance_threshold):
        selected_features = feature_importances >= feature_importance_threshold
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        feature_importances = feature_importances[selected_features]
        feature_names = feature_names[selected_features]

        logging.info(f"递归去除不重要的特征后，剩余特征数: {len(feature_importances)}")

        # 重新训练模型
        random_search.fit(X_train, y_train)
        best_estimator = random_search.best_estimator_
        y_prob = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, 'predict_proba') else best_estimator.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred = (y_prob >= optimal_threshold).astype(int)

        logging.info(f"递归去除不重要的特征后，Stacking模型最佳参数: {random_search.best_params_}")
        logging.info(f"递归去除不重要的特征后，Stacking模型最优阈值: {optimal_threshold}")
        logging.info(f"递归去除不重要的特征后，Stacking模型性能报告：\n{classification_report(y_test, y_pred)}")
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"递归去除不重要的特征后，Stacking模型准确率：{accuracy:.2f}")

        # 更新LightGBM特征重要性
        lgbm_model = best_estimator.named_steps['classifier'].named_estimators_['LightGBM']
        feature_importances = lgbm_model.feature_importances_
        feature_names = np.array([f"feature_{i}" for i in range(X_train.shape[1])])

    # 保存模型、scaler、pca 和 feature_selection
    joblib.dump(best_estimator, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(pca, 'pca.joblib')
    joblib.dump(best_estimator.named_steps['feature_selection'], 'feature_selection.joblib')
    logging.info("模型、scaler、pca 和 feature_selection 已保存")

    return best_estimator, accuracy, y_test, y_pred

# 保存模型
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

    virus_samples_dir = config.get("virus_samples_dir", "E:\\样本库\\待拉黑")
    benign_samples_dir = config.get("benign_samples_dir", "E:\\样本库\\待加入白名单")
    features_path = config.get("features_path", "features.npy")
    labels_path = config.get("labels_path", "labels.npy")
    model_output_path = config.get("model_output_path", "model.joblib")

    start_time = time.time()

    if args.mode == 'incremental':
        logging.info("增量训练模式")
        if os.path.exists(features_path) and os.path.exists(labels_path):
            logging.info(f"特征和标签文件已存在: {features_path}, {labels_path}")
            existing_features, existing_labels = load_features_and_labels(features_path, labels_path)
            if existing_features is None or existing_labels is None:
                logging.error("无法加载特征和标签，重新生成特征和标签...")
                existing_features, existing_labels = [], []
        else:
            existing_features, existing_labels = [], []

        virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir)
        new_features, new_labels = merge_features_and_labels(virus_features, benign_features)

        # 检查新生成的特征和标签数量是否为0
        if len(new_features) == 0 and len(new_labels) == 0:
            logging.info("新生成的特征和标签数量为0，跳过合并数据集，直接使用现有数据集开始训练")
            features = existing_features
            labels = existing_labels
        else:
            if len(existing_features) > 0 and len(existing_labels) > 0:
                features = np.vstack((existing_features, new_features))
                labels = np.concatenate((existing_labels, new_labels))
            else:
                features = new_features
                labels = new_labels

            logging.info(f"新生成的特征和标签已合并到现有数据集中，总样本数: {len(features)}")

        save_features_and_labels(features, labels, features_path, labels_path)

        features, labels = augment_data(features, labels)
        model, accuracy, y_test, y_pred = train_model(features, labels)
        save_model(model, model_output_path)
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

                # 检查新生成的特征和标签数量是否为0
                if len(new_features) == 0 and len(new_labels) == 0:
                    logging.info("新生成的特征和标签数量为0，跳过合并数据集，直接使用现有数据集开始训练")
                    features = existing_features
                    labels = existing_labels
                else:
                    features = np.vstack((existing_features, new_features))
                    labels = np.concatenate((existing_labels, new_labels))
                    logging.info(f"新生成的特征和标签已合并到现有数据集中，总样本数: {len(features)}")

                save_features_and_labels(features, labels, features_path, labels_path)
        else:
            virus_features, benign_features = create_dataset(virus_samples_dir, benign_samples_dir)
            features, labels = merge_features_and_labels(virus_features, benign_features)

            save_features_and_labels(features, labels, features_path, labels_path)

        features, labels = augment_data(features, labels)
        initial_model = joblib.load(model_output_path)
        model, accuracy, y_test, y_pred = train_model(features, labels, initial_model)
        save_model(model, model_output_path)

    end_time = time.time()
    logging.info(f"训练完成，总耗时: {end_time - start_time:.2f} 秒")
