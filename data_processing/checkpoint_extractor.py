import os
import pandas as pd
import numpy as np
import pickle
import json
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import shutil

from feature_extraction import (
    extract_pe_header_features,
    extract_section_features,
    extract_api_features,
    extract_string_features,
    extract_entropy_features,
    extract_api_sequence,
    extract_byte_sequences,
    generate_pe_images
)

class CheckpointExtractor:
    """
    带保存点的数据集提取工具
    支持在特征提取过程中断时从上次停止的地方继续
    """
    def __init__(self, 
                 checkpoint_dir,
                 save_interval=50,  # 每处理多少文件保存一次
                 extract_api_sequences=True,
                 extract_bytes=True,
                 extract_images=True,
                 byte_sequence_length=2048,
                 image_width=32,
                 image_max_bytes=4096,
                 n_jobs=4):
        """
        初始化提取器
        
        Args:
            checkpoint_dir: 保存点目录
            save_interval: 保存间隔（文件数）
            extract_api_sequences: 是否提取API序列
            extract_bytes: 是否提取字节序列
            extract_images: 是否生成图像特征
            byte_sequence_length: 字节序列长度
            image_width: 图像宽度
            image_max_bytes: 图像最大字节数
            n_jobs: 并行作业数
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.extract_api_sequences = extract_api_sequences
        self.extract_bytes = extract_bytes
        self.extract_images = extract_images
        self.byte_sequence_length = byte_sequence_length
        self.image_width = image_width
        self.image_max_bytes = image_max_bytes
        self.n_jobs = n_jobs
        
        # 创建保存点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存点文件路径
        self.checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.json')
        self.features_file = os.path.join(checkpoint_dir, 'features.pkl')
        self.api_sequences_file = os.path.join(checkpoint_dir, 'api_sequences.pkl')
        self.byte_sequences_file = os.path.join(checkpoint_dir, 'byte_sequences.npy')
        self.images_file = os.path.join(checkpoint_dir, 'images.npy')
        
        # 初始化状态
        self.processed_files = set()
        self.features = []
        self.labels = []
        self.api_sequences = []
        self.byte_sequences = []
        self.images = []
        self.file_map = {}  # 文件路径到索引的映射
        
        # 加载检查点（如果存在）
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """加载检查点"""
        if os.path.exists(self.checkpoint_file):
            try:
                # 加载处理记录
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.processed_files = set(checkpoint_data['processed_files'])
                    self.file_map = checkpoint_data['file_map']
                
                # 加载特征数据
                if os.path.exists(self.features_file):
                    with open(self.features_file, 'rb') as f:
                        data = pickle.load(f)
                        self.features = data['features']
                        self.labels = data['labels']
                
                # 加载API序列
                if self.extract_api_sequences and os.path.exists(self.api_sequences_file):
                    with open(self.api_sequences_file, 'rb') as f:
                        self.api_sequences = pickle.load(f)
                
                # 加载字节序列
                if self.extract_bytes and os.path.exists(self.byte_sequences_file):
                    self.byte_sequences = np.load(self.byte_sequences_file, allow_pickle=True)
                
                # 加载图像特征
                if self.extract_images and os.path.exists(self.images_file):
                    self.images = np.load(self.images_file, allow_pickle=True)
                
                print(f"已从检查点恢复，已处理文件数: {len(self.processed_files)}")
                
            except Exception as e:
                print(f"加载检查点出错: {str(e)}")
                print("将重新开始提取")
                self._initialize_extraction()
        else:
            self._initialize_extraction()
    
    def _initialize_extraction(self):
        """初始化提取状态"""
        self.processed_files = set()
        self.features = []
        self.labels = []
        self.api_sequences = []
        self.byte_sequences = []
        self.images = []
        self.file_map = {}
    
    def _save_checkpoint(self):
        """保存检查点"""
        try:
            # 保存之前先创建备份
            if os.path.exists(self.checkpoint_file):
                backup_file = f"{self.checkpoint_file}.bak"
                shutil.copy2(self.checkpoint_file, backup_file)
            
            # 保存处理记录
            checkpoint_data = {
                'processed_files': list(self.processed_files),
                'file_map': self.file_map,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            
            # 保存特征数据
            with open(self.features_file, 'wb') as f:
                data = {'features': self.features, 'labels': self.labels}
                pickle.dump(data, f)
            
            # 保存API序列
            if self.extract_api_sequences and self.api_sequences:
                with open(self.api_sequences_file, 'wb') as f:
                    pickle.dump(self.api_sequences, f)
            
            # 保存字节序列
            if self.extract_bytes and len(self.byte_sequences) > 0:
                np.save(self.byte_sequences_file, self.byte_sequences)
            
            # 保存图像特征
            if self.extract_images and len(self.images) > 0:
                np.save(self.images_file, self.images)
            
            print(f"检查点已保存，已处理文件数: {len(self.processed_files)}")
            
        except Exception as e:
            print(f"保存检查点出错: {str(e)}")
    
    def extract_features_from_file(self, file_path, label):
        """
        从文件提取所有特征
        
        Args:
            file_path: 文件路径
            label: 标签
            
        Returns:
            tuple: (基础特征, API序列, 字节序列, 图像)
        """
        # 如果已经处理过，跳过
        if file_path in self.processed_files:
            return None
        
        try:
            # 提取静态特征
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
            
            # API序列
            api_sequence = None
            if self.extract_api_sequences:
                api_sequence = extract_api_sequence(file_path)
            
            # 字节序列
            byte_sequence = None
            if self.extract_bytes:
                from feature_extraction.byte_sequence import extract_bytes_from_file
                byte_sequence = extract_bytes_from_file(file_path, self.byte_sequence_length)
            
            # 图像特征
            image = None
            if self.extract_images:
                from feature_extraction.image_features import generate_pe_image
                image = generate_pe_image(file_path, self.image_width, self.image_max_bytes)
            
            return (features, api_sequence, byte_sequence, image, label)
            
        except Exception as e:
            print(f"从文件 {file_path} 提取特征时出错: {str(e)}")
            return None
    
    def extract_features_batch(self, file_batch, label):
        """处理一批文件"""
        batch_results = []
        
        for file_path in file_batch:
            # 如果已经处理过，跳过
            if file_path in self.processed_files:
                continue
                
            result = self.extract_features_from_file(file_path, label)
            if result is not None:
                batch_results.append((file_path, result))
        
        return batch_results
    
    def extract_dataset(self, benign_dir, malware_dir, limit=None, shuffle=True):
        """
        提取数据集特征
        
        Args:
            benign_dir: 良性样本目录
            malware_dir: 恶意样本目录
            limit: 每类样本的数量限制
            shuffle: 是否打乱文件顺序
        
        Returns:
            tuple: (X, y, api_sequences, byte_sequences, images)
        """
        # 收集文件路径
        benign_files = []
        for root, _, files in os.walk(benign_dir):
            for file in files:
                if file.lower().endswith(('.exe', '.dll', '.sys')):
                    file_path = os.path.join(root, file)
                    benign_files.append(file_path)
        
        malware_files = []
        for root, _, files in os.walk(malware_dir):
            for file in files:
                if file.lower().endswith(('.exe', '.dll', '.sys')):
                    file_path = os.path.join(root, file)
                    malware_files.append(file_path)
        
        # 应用限制
        if limit:
            benign_files = benign_files[:limit]
            malware_files = malware_files[:limit]
        
        # 打印数据集信息
        print(f"良性样本总数: {len(benign_files)}")
        print(f"恶意样本总数: {len(malware_files)}")
        print(f"已处理文件数: {len(self.processed_files)}")
        
        # 筛选未处理的文件
        benign_files = [f for f in benign_files if f not in self.processed_files]
        malware_files = [f for f in malware_files if f not in self.processed_files]
        
        print(f"待处理良性样本数: {len(benign_files)}")
        print(f"待处理恶意样本数: {len(malware_files)}")
        
        if len(benign_files) == 0 and len(malware_files) == 0:
            print("所有文件已处理完毕")
            return self._finalize_dataset()
        
        # 分批处理
        batch_size = min(50, self.save_interval)
        file_batches = []
        
        # 良性样本分批
        for i in range(0, len(benign_files), batch_size):
            file_batches.append((benign_files[i:i+batch_size], 0))
        
        # 恶意样本分批
        for i in range(0, len(malware_files), batch_size):
            file_batches.append((malware_files[i:i+batch_size], 1))
        
        # 打乱批次顺序
        if shuffle:
            import random
            random.shuffle(file_batches)
        
        # 计算总文件数
        total_files = len(benign_files) + len(malware_files)
        processed_count = 0
        checkpoint_count = 0
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.extract_features_batch, batch, label) for batch, label in file_batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理文件批次"):
                batch_results = future.result()
                
                for file_path, result in batch_results:
                    features, api_sequence, byte_sequence, image, label = result
                    
                    # 更新索引
                    file_index = len(self.features)
                    self.file_map[file_path] = file_index
                    
                    # 添加特征和标签
                    self.features.append(features)
                    self.labels.append(label)
                    
                    # 添加API序列
                    if self.extract_api_sequences:
                        self.api_sequences.append(api_sequence)
                    
                    # 添加字节序列
                    if self.extract_bytes and byte_sequence is not None:
                        if len(self.byte_sequences) == 0:
                            self.byte_sequences = np.array([byte_sequence])
                        else:
                            self.byte_sequences = np.vstack([self.byte_sequences, [byte_sequence]])
                    
                    # 添加图像特征
                    if self.extract_images and image is not None:
                        if len(self.images) == 0:
                            self.images = np.array([image])
                        else:
                            # 检查图像高度是否一致
                            if image.shape[0] != self.images.shape[1]:
                                # 调整图像高度
                                if image.shape[0] > self.images.shape[1]:
                                    # 调整之前的图像
                                    new_images = np.zeros((len(self.images), image.shape[0], 
                                                          self.images.shape[2], self.images.shape[3]), 
                                                          dtype=self.images.dtype)
                                    new_images[:, :self.images.shape[1], :, :] = self.images
                                    self.images = new_images
                                else:
                                    # 调整当前图像
                                    new_image = np.zeros((1, self.images.shape[1], 
                                                         image.shape[1], image.shape[2]), 
                                                         dtype=image.dtype)
                                    new_image[0, :image.shape[0], :, :] = image
                                    image = new_image[0]
                            
                            self.images = np.vstack([self.images, [image]])
                    
                    # 标记为已处理
                    self.processed_files.add(file_path)
                    
                    processed_count += 1
                    checkpoint_count += 1
                
                # 检查是否需要保存检查点
                if checkpoint_count >= self.save_interval:
                    self._save_checkpoint()
                    checkpoint_count = 0
                    
                    # 打印进度
                    progress = (len(self.processed_files) / (total_files + len(self.processed_files) - processed_count)) * 100
                    print(f"进度: {progress:.2f}% ({len(self.processed_files)} / {total_files + len(self.processed_files) - processed_count})")
        
        # 最终保存
        self._save_checkpoint()
        
        return self._finalize_dataset()
    
    def _finalize_dataset(self):
        """完成数据集处理并返回结果"""
        # 转换为DataFrame
        X = pd.DataFrame(self.features)
        y = np.array(self.labels)
        
        # 确保没有NaN值
        X = X.fillna(0)
        
        # 返回结果
        if self.extract_api_sequences and self.extract_bytes and self.extract_images:
            return X, y, self.api_sequences, self.byte_sequences, self.images
        elif self.extract_api_sequences and self.extract_bytes:
            return X, y, self.api_sequences, self.byte_sequences, None
        elif self.extract_api_sequences and self.extract_images:
            return X, y, self.api_sequences, None, self.images
        elif self.extract_bytes and self.extract_images:
            return X, y, None, self.byte_sequences, self.images
        elif self.extract_api_sequences:
            return X, y, self.api_sequences, None, None
        elif self.extract_bytes:
            return X, y, None, self.byte_sequences, None
        elif self.extract_images:
            return X, y, None, None, self.images
        else:
            return X, y, None, None, None 