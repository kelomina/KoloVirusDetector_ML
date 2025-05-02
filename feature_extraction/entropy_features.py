import pefile
import numpy as np
import math
from collections import Counter

def calculate_shannon_entropy(data):
    """
    计算字节序列的香农熵
    
    Args:
        data: 字节序列
        
    Returns:
        float: 香农熵值
    """
    if not data:
        return 0
    
    counter = Counter(data)
    total = len(data)
    entropy = 0
    
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_mean_entropy(data, window_size=256):
    """
    计算均值熵，使用滑动窗口计算局部熵的均值
    
    Args:
        data: 字节序列
        window_size: 滑动窗口大小
        
    Returns:
        float: 均值熵
    """
    if not data or len(data) < window_size:
        return calculate_shannon_entropy(data)
    
    entropies = []
    for i in range(0, len(data) - window_size + 1, window_size // 2):  # 步长为窗口的一半，实现50%重叠
        window = data[i:i+window_size]
        entropies.append(calculate_shannon_entropy(window))
    
    return np.mean(entropies) if entropies else 0

def calculate_variance_entropy(data, window_size=256):
    """
    计算方差熵，使用滑动窗口计算局部熵的方差
    
    Args:
        data: 字节序列
        window_size: 滑动窗口大小
        
    Returns:
        float: 方差熵
    """
    if not data or len(data) < window_size:
        return 0
    
    entropies = []
    for i in range(0, len(data) - window_size + 1, window_size // 2):  # 步长为窗口的一半，实现50%重叠
        window = data[i:i+window_size]
        entropies.append(calculate_shannon_entropy(window))
    
    return np.var(entropies) if entropies else 0

def extract_entropy_features(file_path):
    """
    提取PE文件的熵值特征矩阵
    
    Args:
        file_path: PE文件路径
        
    Returns:
        dict: 熵值特征字典
    """
    try:
        pe = pefile.PE(file_path)
        features = {}
        
        # 整个文件的熵值
        full_data = pe.__data__
        features['file_shannon_entropy'] = calculate_shannon_entropy(full_data)
        features['file_mean_entropy'] = calculate_mean_entropy(full_data)
        features['file_variance_entropy'] = calculate_variance_entropy(full_data)
        
        # 计算各节区的熵值
        important_sections = ['.text', '.data', '.rdata', '.rsrc', '.reloc']
        section_entropies = {}
        
        for section in pe.sections:
            section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
            
            # 获取节区数据
            section_data = pe.__data__[section.PointerToRawData:section.PointerToRawData+section.SizeOfRawData]
            
            # 如果节区长度过短，跳过
            if len(section_data) <= 10:
                continue
                
            # 计算该节区的三种熵值
            section_entropies[section_name] = {
                'shannon': calculate_shannon_entropy(section_data),
                'mean': calculate_mean_entropy(section_data),
                'variance': calculate_variance_entropy(section_data)
            }
            
            # 添加节区熵值特征
            features[f'entropy_{section_name}_shannon'] = section_entropies[section_name]['shannon']
            features[f'entropy_{section_name}_mean'] = section_entropies[section_name]['mean']
            features[f'entropy_{section_name}_variance'] = section_entropies[section_name]['variance']
        
        # 对于找不到的重要节区，设置默认值
        for section_name in important_sections:
            if section_name not in section_entropies:
                features[f'entropy_{section_name}_shannon'] = 0
                features[f'entropy_{section_name}_mean'] = 0
                features[f'entropy_{section_name}_variance'] = 0
        
        # 计算节区熵值统计量
        if section_entropies:
            shannon_values = [values['shannon'] for values in section_entropies.values()]
            mean_values = [values['mean'] for values in section_entropies.values()]
            variance_values = [values['variance'] for values in section_entropies.values()]
            
            # 熵值的统计特征
            features['section_entropy_shannon_min'] = min(shannon_values)
            features['section_entropy_shannon_max'] = max(shannon_values)
            features['section_entropy_shannon_mean'] = np.mean(shannon_values)
            features['section_entropy_shannon_std'] = np.std(shannon_values)
            
            features['section_entropy_mean_min'] = min(mean_values)
            features['section_entropy_mean_max'] = max(mean_values)
            features['section_entropy_mean_mean'] = np.mean(mean_values)
            features['section_entropy_mean_std'] = np.std(mean_values)
            
            features['section_entropy_variance_min'] = min(variance_values)
            features['section_entropy_variance_max'] = max(variance_values)
            features['section_entropy_variance_mean'] = np.mean(variance_values)
            features['section_entropy_variance_std'] = np.std(variance_values)
        else:
            # 没有有效节区时的默认值
            for stat in ['min', 'max', 'mean', 'std']:
                features[f'section_entropy_shannon_{stat}'] = 0
                features[f'section_entropy_mean_{stat}'] = 0
                features[f'section_entropy_variance_{stat}'] = 0
        
        # 计算特征交叉：最高熵节区与最低熵节区的比值
        if section_entropies and len(shannon_values) > 1:
            features['entropy_ratio_max_min'] = max(shannon_values) / (min(shannon_values) + 1e-10)
            
            # 代码节区和数据节区的熵值比
            if '.text' in section_entropies and '.data' in section_entropies:
                features['entropy_ratio_text_data'] = section_entropies['.text']['shannon'] / (section_entropies['.data']['shannon'] + 1e-10)
            else:
                features['entropy_ratio_text_data'] = 0
        else:
            features['entropy_ratio_max_min'] = 0
            features['entropy_ratio_text_data'] = 0
            
        # 熵值与文件大小的相关性
        features['entropy_file_size_ratio'] = features['file_shannon_entropy'] / (len(full_data) + 1e-10)
        
        return features
    
    except Exception as e:
        print(f"提取熵值特征时出错 {file_path}: {str(e)}")
        return {} 