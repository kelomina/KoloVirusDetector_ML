import os
import numpy as np
from tqdm import tqdm

def extract_bytes_from_file(file_path, max_length=2048):
    """
    从文件中提取原始字节序列
    
    Args:
        file_path: 文件路径
        max_length: 最大序列长度
        
    Returns:
        numpy array: 字节序列，长度为max_length，不足部分用0填充
    """
    try:
        # 读取文件的二进制内容
        with open(file_path, 'rb') as f:
            byte_content = f.read()
            
        # 转换为numpy数组，限制长度并填充
        byte_array = np.frombuffer(byte_content, dtype=np.uint8)
        if len(byte_array) > max_length:
            byte_array = byte_array[:max_length]
        else:
            # 填充0
            byte_array = np.pad(byte_array, (0, max_length - len(byte_array)), 'constant')
            
        # 归一化到[0,1]范围
        byte_array = byte_array.astype(np.float32) / 255.0
        
        return byte_array
        
    except Exception as e:
        print(f"从文件提取字节序列时出错 {file_path}: {str(e)}")
        # 返回全0数组
        return np.zeros(max_length, dtype=np.float32)
    
def extract_byte_sequences(file_paths, max_length=2048):
    """
    从多个文件中提取字节序列
    
    Args:
        file_paths: 文件路径列表
        max_length: 每个序列的最大长度
        
    Returns:
        numpy array: 形状为[n_samples, max_length]的字节序列数组
    """
    all_sequences = []
    
    for file_path in tqdm(file_paths, desc="提取字节序列"):
        sequence = extract_bytes_from_file(file_path, max_length)
        all_sequences.append(sequence)
    
    return np.array(all_sequences) 