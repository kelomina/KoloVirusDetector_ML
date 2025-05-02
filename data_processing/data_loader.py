import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from feature_extraction import (
    extract_pe_header_features,
    extract_section_features,
    extract_api_features,
    extract_string_features,
    extract_entropy_features
)

def extract_all_features(file_path):
    """
    从PE文件中提取所有特征
    
    Args:
        file_path: PE文件路径
        
    Returns:
        dict: 所有特征的合并字典
    """
    try:
        # 提取各类特征
        features = {}
        
        # PE头特征
        pe_header_features = extract_pe_header_features(file_path)
        features.update(pe_header_features)
        
        # 节区特征
        section_features = extract_section_features(file_path)
        features.update(section_features)
        
        # API调用特征
        api_features = extract_api_features(file_path)
        features.update(api_features)
        
        # 字符串特征
        string_features = extract_string_features(file_path)
        features.update(string_features)
        
        # 熵值特征
        entropy_features = extract_entropy_features(file_path)
        features.update(entropy_features)
        
        return features
    
    except Exception as e:
        print(f"提取所有特征时出错 {file_path}: {str(e)}")
        return None

def load_dataset(benign_dir, malware_dir, output_file=None, n_jobs=4, limit=None):
    """
    加载良性和恶意PE文件并提取特征
    
    Args:
        benign_dir: 良性PE文件目录
        malware_dir: 恶意PE文件目录
        output_file: 输出特征数据的文件路径
        n_jobs: 并行作业数
        limit: 每类样本的最大数量限制
        
    Returns:
        tuple: (X, y) 特征矩阵和标签
    """
    # 收集文件路径
    benign_files = []
    for root, _, files in os.walk(benign_dir):
        for file in files:
            if file.lower().endswith(('.exe', '.dll', '.sys')):
                benign_files.append(os.path.join(root, file))
    
    malware_files = []
    for root, _, files in os.walk(malware_dir):
        for file in files:
            if file.lower().endswith(('.exe', '.dll', '.sys')):
                malware_files.append(os.path.join(root, file))
    
    # 应用限制
    if limit:
        benign_files = benign_files[:limit]
        malware_files = malware_files[:limit]
    
    # 打印数据集信息
    print(f"良性样本数量: {len(benign_files)}")
    print(f"恶意样本数量: {len(malware_files)}")
    
    # 并行提取特征
    all_features = []
    all_labels = []
    
    def process_file_batch(file_batch, label):
        batch_features = []
        batch_labels = []
        
        for file_path in file_batch:
            features = extract_all_features(file_path)
            if features is not None:
                batch_features.append(features)
                batch_labels.append(label)
        
        return batch_features, batch_labels
    
    # 分批处理
    batch_size = 50
    file_batches = []
    
    # 良性样本分批
    for i in range(0, len(benign_files), batch_size):
        file_batches.append((benign_files[i:i+batch_size], 0))
    
    # 恶意样本分批
    for i in range(0, len(malware_files), batch_size):
        file_batches.append((malware_files[i:i+batch_size], 1))
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_file_batch, batch, label) for batch, label in file_batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理文件"):
            batch_features, batch_labels = future.result()
            all_features.extend(batch_features)
            all_labels.extend(batch_labels)
    
    # 转换为DataFrame
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    
    # 确保没有NaN值
    X = X.fillna(0)
    
    # 保存结果
    if output_file:
        data = {'X': X, 'y': y}
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"特征数据已保存到: {output_file}")
    
    return X, y

def load_single_file(file_path):
    """
    从单个PE文件提取特征
    
    Args:
        file_path: PE文件路径
        
    Returns:
        pd.DataFrame: 特征DataFrame
    """
    features = extract_all_features(file_path)
    if features is None:
        return None
    
    # 转换为DataFrame
    X = pd.DataFrame([features])
    
    # 确保没有NaN值
    X = X.fillna(0)
    
    return X 