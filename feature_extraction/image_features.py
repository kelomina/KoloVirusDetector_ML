import os
import numpy as np
from tqdm import tqdm
import math

def bytes_to_image(byte_array, width=32):
    """
    将字节数组转换为2D图像
    
    Args:
        byte_array: 字节数组
        width: 图像宽度
        
    Returns:
        numpy array: 形状为[height, width, 1]的图像
    """
    # 计算所需的图像高度
    height = math.ceil(len(byte_array) / width)
    
    # 创建一个全零的图像
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # 填充图像
    for i in range(len(byte_array)):
        row = i // width
        col = i % width
        
        if row < height and col < width:
            # 根据字节值计算RGB通道值
            # R通道 - 基于字节值的高4位
            image[row, col, 0] = (byte_array[i] & 0xF0) / 240.0
            # G通道 - 基于字节值的低4位
            image[row, col, 1] = (byte_array[i] & 0x0F) / 15.0
            # B通道 - 基于字节值的奇偶性和其他特性
            image[row, col, 2] = 1.0 if byte_array[i] % 2 == 0 else 0.0
    
    return image

def generate_pe_image(file_path, width=32, max_bytes=4096):
    """
    从PE文件生成图像表示
    
    Args:
        file_path: PE文件路径
        width: 图像宽度
        max_bytes: 最大处理的字节数
        
    Returns:
        numpy array: 形状为[height, width, 3]的图像
    """
    try:
        # 读取文件的二进制内容
        with open(file_path, 'rb') as f:
            content = f.read(max_bytes)
            byte_array = np.frombuffer(content, dtype=np.uint8)
        
        # 转换为图像
        image = bytes_to_image(byte_array, width)
        return image
        
    except Exception as e:
        print(f"从文件生成图像时出错 {file_path}: {str(e)}")
        # 返回一个空图像
        height = math.ceil(max_bytes / width)
        return np.zeros((height, width, 3), dtype=np.float32)

def generate_pe_images(file_paths, width=32, max_bytes=4096):
    """
    从多个PE文件生成图像表示
    
    Args:
        file_paths: PE文件路径列表
        width: 图像宽度
        max_bytes: 最大处理的字节数
        
    Returns:
        numpy array: 形状为[n_samples, height, width, 3]的图像数组
    """
    all_images = []
    
    for file_path in tqdm(file_paths, desc="生成PE图像"):
        image = generate_pe_image(file_path, width, max_bytes)
        all_images.append(image)
    
    # 确保所有图像具有相同的高度
    max_height = max(img.shape[0] for img in all_images)
    normalized_images = []
    
    for img in all_images:
        if img.shape[0] < max_height:
            # 填充高度不足的图像
            padded_img = np.zeros((max_height, width, 3), dtype=np.float32)
            padded_img[:img.shape[0], :, :] = img
            normalized_images.append(padded_img)
        else:
            normalized_images.append(img)
    
    return np.array(normalized_images) 