import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
from torch.cuda import amp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from models.lightgbm_model import MalwareDetector as LGBMDetector
import joblib

class ResidualBlock(nn.Module):
    """
    残差块 - 用于PE头特征处理
    """
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # 残差连接
        out = F.relu(out)
        return out

class AttentionLayer(nn.Module):
    """
    时间注意力层 - 为API序列分配重要性权重
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output, api_importance=None):
        # lstm_output形状: [batch_size, seq_len, hidden_dim]
        
        # 计算注意力权重
        attention_weights = self.attention(lstm_output).squeeze(2)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
        
        # 如果提供了API重要性，将其融入注意力权重
        if api_importance is not None:
            # api_importance形状: [batch_size, seq_len]
            # 将API重要性与注意力权重相乘
            attention_weights = attention_weights * api_importance
            # 重新归一化
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
        
        # 注意力加权和
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # [batch_size, hidden_dim]
        
        return context_vector, attention_weights

class APICallPatternAttention(nn.Module):
    """
    可解释的API调用模式关注机制 - 识别恶意软件中的关键API调用模式
    """
    def __init__(self, hidden_dim, num_patterns=8, pattern_size=5):
        super(APICallPatternAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.pattern_size = pattern_size
        
        # 可学习的API调用模式
        self.patterns = nn.Parameter(
            torch.randn(num_patterns, hidden_dim),
            requires_grad=True
        )
        
        # 模式重要性权重 - 每个模式对恶意软件检测的重要性
        self.pattern_importance = nn.Parameter(
            torch.ones(num_patterns),
            requires_grad=True
        )
        
        # 模式描述符 - 用于解释每个模式的语义含义
        self.pattern_query = nn.Linear(hidden_dim, num_patterns)
        
        # 局部注意力层 - 捕捉局部上下文信息
        self.local_attention = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=pattern_size,
            padding=(pattern_size-1)//2
        )
        
        # 全局注意力层 - 将局部注意力信息与整体信息结合
        self.global_attention = nn.Linear(hidden_dim, 1)
        
    def get_pattern_names(self, tokenizer=None, top_k=5):
        """
        生成可解释的模式名称，基于每个模式最相似的API调用
        
        Args:
            tokenizer: APITokenizer实例，用于将ID映射回API名称
            top_k: 每个模式返回的最相似API数量
            
        Returns:
            pattern_names: 模式名称列表
        """
        if tokenizer is None or not hasattr(tokenizer, 'encoder'):
            return [f"Pattern-{i}" for i in range(self.num_patterns)]
        
        pattern_names = []
        # 获取所有API的嵌入
        all_apis = tokenizer.encoder.classes_
        
        # 排除特殊标记
        valid_apis = [api for api in all_apis if api not in ['<PAD>', '<UNK>']]
        
        # 为每个模式生成名称
        for pattern_idx in range(self.num_patterns):
            pattern_vec = self.patterns[pattern_idx].detach().cpu().numpy()
            
            # 计算每个API与该模式的相似度
            api_scores = []
            for api in valid_apis:
                try:
                    api_id = tokenizer.encoder.transform([api])[0]
                    # 这里假设模型的嵌入层是可访问的
                    # 实际使用时需要根据模型结构进行适配
                    api_score = pattern_vec.dot(pattern_vec) / (np.linalg.norm(pattern_vec) * np.linalg.norm(pattern_vec))
                    api_scores.append((api, api_score))
                except:
                    continue
            
            # 获取得分最高的API
            api_scores.sort(key=lambda x: x[1], reverse=True)
            top_apis = [api for api, _ in api_scores[:top_k]]
            
            # 生成模式名称
            pattern_name = f"Pattern-{pattern_idx}: {', '.join(top_apis)}"
            pattern_names.append(pattern_name)
            
        return pattern_names
        
    def forward(self, lstm_output, api_importance=None, return_pattern_scores=False):
        """
        前向传播
        
        Args:
            lstm_output: LSTM层输出，形状为[batch_size, seq_len, hidden_dim]
            api_importance: API重要性权重，形状为[batch_size, seq_len]
            return_pattern_scores: 是否返回模式激活分数
            
        Returns:
            context_vector: 上下文向量
            attention_weights: 注意力权重
            pattern_activations: (可选) 模式激活分数
        """
        batch_size, seq_len, hidden_dim = lstm_output.shape
        
        # 局部特征提取
        # 转换为卷积层所需的输入形状[batch_size, hidden_dim, seq_len]
        local_input = lstm_output.transpose(1, 2)
        local_features = self.local_attention(local_input)
        # 转换回[batch_size, seq_len, hidden_dim]
        local_features = local_features.transpose(1, 2)[:, :seq_len, :]
        
        # 模式匹配 - 计算每个位置与每个模式的相似度
        # 扩展模式为[num_patterns, 1, hidden_dim]以便广播
        patterns_expanded = self.patterns.unsqueeze(1)
        # lstm_output形状为[batch_size, seq_len, hidden_dim]
        
        # 计算每个模式在每个位置的激活分数
        # 结果形状为[batch_size, seq_len, num_patterns]
        pattern_scores = torch.matmul(lstm_output, patterns_expanded.transpose(1, 2))
        
        # 应用模式重要性权重
        # pattern_importance形状为[num_patterns]
        # 扩展为[1, 1, num_patterns]以便广播
        weighted_pattern_scores = pattern_scores * self.pattern_importance.view(1, 1, -1)
        
        # 汇总每个位置的模式激活得分
        # 结果形状为[batch_size, seq_len]
        pattern_activations = torch.sum(weighted_pattern_scores, dim=2)
        
        # 全局注意力
        global_scores = self.global_attention(lstm_output).squeeze(2)  # [batch_size, seq_len]
        
        # 结合局部模式激活和全局注意力
        combined_scores = pattern_activations + global_scores
        
        # 应用softmax获得注意力权重
        attention_weights = F.softmax(combined_scores, dim=1)  # [batch_size, seq_len]
        
        # 如果提供了API重要性，将其融入注意力权重
        if api_importance is not None:
            # 将API重要性与注意力权重相乘
            attention_weights = attention_weights * api_importance
            # 重新归一化
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
        
        # 注意力加权和
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # [batch_size, hidden_dim]
        
        if return_pattern_scores:
            return context_vector, attention_weights, weighted_pattern_scores
        
        return context_vector, attention_weights

class BiLSTMAttention(nn.Module):
    """
    具有注意力机制的双向LSTM模型 - 用于处理API调用序列
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.3, use_pattern_attention=True):
        super(BiLSTMAttention, self).__init__()
        
        self.use_pattern_attention = use_pattern_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
                            
        # 使用可解释的API调用模式关注机制或普通注意力
        if use_pattern_attention:
            self.attention = APICallPatternAttention(hidden_dim * 2)  # 双向LSTM输出维度是hidden_dim*2
        else:
            self.attention = AttentionLayer(hidden_dim * 2)  # 双向LSTM输出维度是hidden_dim*2
            
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, api_sequences, api_importance=None, sequence_lengths=None):
        # api_sequences形状: [batch_size, seq_len]
        
        # 嵌入层
        embedded = self.dropout(self.embedding(api_sequences))  # [batch_size, seq_len, embedding_dim]
        
        # 如果提供了序列长度，使用pack_padded_sequence
        if sequence_lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths, 
                                                              batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            # 解包
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # 使用注意力机制获取上下文向量
        if self.use_pattern_attention:
            context, attention_weights, pattern_scores = self.attention(lstm_output, api_importance, return_pattern_scores=True)
        else:
            context, attention_weights = self.attention(lstm_output, api_importance)
            pattern_scores = None
        
        # 最终输出
        output = self.fc(context)
        
        if self.use_pattern_attention:
            return output, attention_weights, context, pattern_scores
        else:
            return output, attention_weights, context

class ByteSeqCNN(nn.Module):
    """
    字节序列卷积网络 - 处理PE文件的原始字节序列
    """
    def __init__(self, input_channels=1, output_dim=128, dropout=0.3):
        super(ByteSeqCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # x形状: [batch_size, seq_len] -> [batch_size, 1, seq_len]
        x = x.unsqueeze(1)
        
        # 卷积层
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class ImageCNN(nn.Module):
    """
    图像卷积网络 - 处理PE文件的可视化图像
    """
    def __init__(self, input_channels=3, output_dim=128, dropout=0.3):
        super(ImageCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, 1).squeeze(2).squeeze(2)
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class MultiModalFusion(nn.Module):
    """
    多模态特征融合模块
    """
    def __init__(self, pe_dim, api_dim, byte_dim, img_dim, fusion_dim, dropout=0.3):
        super(MultiModalFusion, self).__init__()
        
        # 模态注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(pe_dim + api_dim + byte_dim + img_dim, 4),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion = nn.Linear(pe_dim + api_dim + byte_dim + img_dim, fusion_dim)
        self.bn_fusion = nn.BatchNorm1d(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pe_features, api_features, byte_features=None, img_features=None):
        # 准备所有可用的特征
        features = [pe_features, api_features]
        if byte_features is not None:
            features.append(byte_features)
        if img_features is not None:
            features.append(img_features)
        
        # 计算模态重要性权重
        combined = torch.cat(features, dim=1)
        attention_weights = self.attention(combined)
        
        # 应用注意力权重
        weighted_features = []
        start_idx = 0
        for i, feature in enumerate(features):
            end_idx = start_idx + feature.size(1)
            weighted_feature = feature * attention_weights[:, i:i+1]
            weighted_features.append(weighted_feature)
            start_idx = end_idx
        
        # 融合特征
        fused = torch.cat(weighted_features, dim=1)
        fused = self.fusion(fused)
        fused = self.bn_fusion(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        return fused, attention_weights

class MalwareDeepModel(nn.Module):
    """
    恶意软件检测多模态深度学习模型 - 融合PE特征、API序列、字节序列和图像特征
    """
    def __init__(self, pe_feature_dim, vocab_size, embedding_dim=128, hidden_dim=128,
                lstm_layers=1, fc_hidden=256, dropout=0.3, use_byte_seq=True, use_image=True):
        super(MalwareDeepModel, self).__init__()
        
        self.use_byte_seq = use_byte_seq
        self.use_image = use_image
        
        # PE头特征处理分支 - 使用残差块
        self.pe_feature_norm = nn.BatchNorm1d(pe_feature_dim)
        self.residual_block1 = ResidualBlock(pe_feature_dim, fc_hidden)
        self.residual_block2 = ResidualBlock(pe_feature_dim, fc_hidden)
        
        # API序列处理分支 - 使用Bi-LSTM和注意力
        self.bilstm_attention = BiLSTMAttention(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=fc_hidden,
            n_layers=lstm_layers,
            dropout=dropout
        )
        
        # 字节序列处理分支 - 使用CNN
        if self.use_byte_seq:
            self.byte_cnn = ByteSeqCNN(
                input_channels=1,
                output_dim=fc_hidden,
                dropout=dropout
            )
        
        # 图像特征处理分支 - 使用CNN
        if self.use_image:
            self.image_cnn = ImageCNN(
                input_channels=3,
                output_dim=fc_hidden,
                dropout=dropout
            )
        
        # 多模态特征融合
        total_dim = pe_feature_dim + fc_hidden
        if self.use_byte_seq:
            total_dim += fc_hidden
        if self.use_image:
            total_dim += fc_hidden
            
        self.fusion = MultiModalFusion(
            pe_dim=pe_feature_dim,
            api_dim=fc_hidden,
            byte_dim=fc_hidden if self.use_byte_seq else 0,
            img_dim=fc_hidden if self.use_image else 0,
            fusion_dim=fc_hidden,
            dropout=dropout
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pe_features, api_sequences, api_importance=None, sequence_lengths=None, 
                byte_sequences=None, images=None):
        # 处理PE头特征
        pe_norm = self.pe_feature_norm(pe_features)
        pe_res1 = self.residual_block1(pe_norm)
        pe_features_processed = self.residual_block2(pe_res1)
        
        # 处理API序列特征
        api_output, attention_weights, api_context, pattern_scores = self.bilstm_attention(
            api_sequences, api_importance, sequence_lengths
        )
        
        # 处理字节序列特征（如果启用）
        byte_features = None
        if self.use_byte_seq and byte_sequences is not None:
            byte_features = self.byte_cnn(byte_sequences)
        
        # 处理图像特征（如果启用）
        img_features = None
        if self.use_image and images is not None:
            img_features = self.image_cnn(images)
        
        # 多模态特征融合
        fused, modal_weights = self.fusion(
            pe_features_processed, api_context, byte_features, img_features
        )
        
        # 最终分类
        output = self.classifier(fused)
        
        return output, attention_weights, modal_weights, pattern_scores

class APITokenizer:
    """
    将API序列转换为模型可用的数值张量
    """
    def __init__(self, max_features=10000, max_length=100):
        self.max_features = max_features
        self.max_length = max_length
        self.encoder = LabelEncoder()
        self.api_importance = {}  # 存储API的重要性得分
        self.fitted = False
        
    def fit(self, api_sequences, api_importance_scores=None):
        """
        拟合编码器，将API名称映射为数字ID
        
        Args:
            api_sequences: API序列列表
            api_importance_scores: API重要性字典 {api_name: importance_score}
        """
        # 确保api_sequences是列表列表形式
        if isinstance(api_sequences[0], str):
            api_sequences = [api_sequences]
            
        # 收集所有不同的API名称
        all_apis = []
        for seq in api_sequences:
            all_apis.extend(seq)
            
        # 增加特殊标记
        all_apis.append('<PAD>')  # 填充标记
        all_apis.append('<UNK>')  # 未知API标记
        
        # 拟合编码器
        self.encoder.fit(all_apis)
        
        # 存储API重要性得分
        if api_importance_scores:
            for api, score in api_importance_scores.items():
                if api in self.encoder.classes_:
                    api_id = self.encoder.transform([api])[0]
                    self.api_importance[api_id] = score
                    
        self.fitted = True
        return self
    
    def transform(self, api_sequences):
        """
        将API序列转换为数字序列
        
        Args:
            api_sequences: API序列列表
            
        Returns:
            transformed_sequences: 编码后的序列
            sequence_lengths: 每个序列的真实长度
            importance_weights: 重要性权重矩阵
        """
        if not self.fitted:
            raise ValueError("模型未拟合，请先调用fit方法")
            
        # 确保api_sequences是列表列表形式
        if isinstance(api_sequences[0], str):
            api_sequences = [api_sequences]
            
        # 获取PAD和UNK的ID
        pad_id = self.encoder.transform(['<PAD>'])[0]
        unk_id = self.encoder.transform(['<UNK>'])[0]
        
        transformed_sequences = []
        sequence_lengths = []
        importance_weights = []
        
        for seq in api_sequences:
            # 截断序列
            seq = seq[:self.max_length]
            seq_len = len(seq)
            sequence_lengths.append(seq_len)
            
            # 转换API为ID
            try:
                seq_ids = self.encoder.transform(seq).tolist()
            except:
                # 处理未知API
                seq_ids = []
                for api in seq:
                    try:
                        api_id = self.encoder.transform([api])[0]
                    except:
                        api_id = unk_id
                    seq_ids.append(api_id)
            
            # 创建序列张量
            transformed_sequences.append(torch.tensor(seq_ids, dtype=torch.long))
            
            # 创建重要性权重
            if self.api_importance:
                weights = [self.api_importance.get(api_id, 1.0) for api_id in seq_ids]
            else:
                weights = [1.0] * seq_len
                
            importance_weights.append(torch.tensor(weights, dtype=torch.float))
            
        # 使用pad_sequence处理变长序列
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            transformed_sequences, 
            batch_first=True, 
            padding_value=pad_id
        )
        
        padded_weights = torch.nn.utils.rnn.pad_sequence(
            importance_weights,
            batch_first=True,
            padding_value=0.0
        )
            
        return padded_sequences.numpy(), np.array(sequence_lengths), padded_weights.numpy()
    
    def fit_transform(self, api_sequences, api_importance_scores=None):
        """
        拟合并转换API序列
        """
        self.fit(api_sequences, api_importance_scores)
        return self.transform(api_sequences)
    
    def get_vocab_size(self):
        """
        返回词汇表大小
        """
        if not self.fitted:
            raise ValueError("模型未拟合，请先调用fit方法")
        return len(self.encoder.classes_)

class MalwareDataset(Dataset):
    """
    恶意软件数据集，用于高效数据加载
    """
    def __init__(self, pe_features, api_sequences, labels, 
                 byte_sequences=None, images=None, tokenizer=None):
        """
        初始化数据集
        
        Args:
            pe_features: PE特征DataFrame
            api_sequences: API序列列表
            labels: 标签数组
            byte_sequences: 字节序列特征(可选)
            images: 图像特征(可选)
            tokenizer: API序列标记器
        """
        self.pe_features = pe_features
        self.api_sequences = api_sequences
        self.labels = labels
        self.byte_sequences = byte_sequences
        self.images = images
        self.tokenizer = tokenizer
        
        # 预处理API序列
        if self.tokenizer:
            self.processed_api_sequences, self.sequence_lengths, self.api_importance = \
                self.tokenizer.transform(self.api_sequences)
        
    def __len__(self):
        return len(self.pe_features)
    
    def __getitem__(self, idx):
        pe_feature = torch.tensor(self.pe_features.iloc[idx].values, dtype=torch.float32)
        
        # API序列
        if self.tokenizer:
            api_seq = torch.tensor(self.processed_api_sequences[idx], dtype=torch.long)
            seq_len = self.sequence_lengths[idx]
            api_imp = torch.tensor(self.api_importance[idx], dtype=torch.float32)
        else:
            api_seq = self.api_sequences[idx]
            seq_len = len(api_seq)
            api_imp = torch.ones(seq_len, dtype=torch.float32)
        
        # 字节序列
        if self.byte_sequences is not None:
            byte_seq = torch.tensor(self.byte_sequences[idx], dtype=torch.float32)
        else:
            byte_seq = None
            
        # 图像
        if self.images is not None:
            image = torch.tensor(self.images[idx], dtype=torch.float32)
        else:
            image = None
            
        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        
        return {
            'pe_feature': pe_feature,
            'api_seq': api_seq,
            'seq_len': seq_len,
            'api_imp': api_imp,
            'byte_seq': byte_seq,
            'image': image,
            'label': label
        }

def collate_fn(batch):
    """
    自定义批处理函数，处理变长序列和可选特征
    """
    pe_features = torch.stack([item['pe_feature'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # API序列相关
    api_seqs = [item['api_seq'] for item in batch]
    seq_lens = [item['seq_len'] for item in batch]
    api_imps = [item['api_imp'] for item in batch]
    
    # 变长序列填充
    api_seqs_padded = torch.nn.utils.rnn.pad_sequence(api_seqs, batch_first=True)
    api_imps_padded = torch.nn.utils.rnn.pad_sequence(api_imps, batch_first=True, padding_value=0.0)
    
    # 字节序列和图像(如果有)
    if batch[0]['byte_seq'] is not None:
        byte_seqs = torch.stack([item['byte_seq'] for item in batch])
    else:
        byte_seqs = None
        
    if batch[0]['image'] is not None:
        images = torch.stack([item['image'] for item in batch])
    else:
        images = None
        
    return {
        'pe_features': pe_features,
        'api_seqs': api_seqs_padded,
        'seq_lens': torch.tensor(seq_lens, dtype=torch.long),
        'api_imps': api_imps_padded,
        'byte_seqs': byte_seqs,
        'images': images,
        'labels': labels
    }

class DeepMalwareDetector:
    """
    深度学习恶意软件检测器 - 使用SWA优化和多模态融合
    """
    def __init__(self, pe_feature_dim, api_sequences=None, api_importance_scores=None,
                embedding_dim=128, hidden_dim=128, lstm_layers=1, fc_hidden=256, 
                dropout=0.3, lr=0.001, batch_size=64, epochs=50, swa_start=5,
                use_byte_seq=True, use_image=True, use_amp=True):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.fc_hidden = fc_hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.swa_start = swa_start
        self.use_byte_seq = use_byte_seq
        self.use_image = use_image
        self.use_amp = use_amp
        
        # 初始化API标记器
        self.tokenizer = APITokenizer()
        if api_sequences:
            self.tokenizer.fit(api_sequences, api_importance_scores)
            vocab_size = self.tokenizer.get_vocab_size()
        else:
            vocab_size = 10000  # 默认值
        
        # 初始化多模态模型
        self.model = MalwareDeepModel(
            pe_feature_dim=pe_feature_dim,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            fc_hidden=fc_hidden,
            dropout=dropout,
            use_byte_seq=use_byte_seq,
            use_image=use_image
        )
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 初始化SWA优化器
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, anneal_epochs=1, swa_lr=lr/2)
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.swa_model.to(self.device)
        
        # 初始化混合精度训练所需的scaler
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = amp.GradScaler()
        else:
            self.use_amp = False
        
    def prepare_batch(self, pe_features, api_sequences=None, labels=None, byte_sequences=None, images=None):
        """
        准备批次数据
        """
        # 处理PE特征
        pe_features_tensor = torch.tensor(pe_features.values, dtype=torch.float32).to(self.device)
        
        # 处理API序列（如果提供）
        if api_sequences is not None:
            sequences, lengths, importance = self.tokenizer.transform(api_sequences)
            api_seq_tensor = torch.tensor(sequences, dtype=torch.long).to(self.device)
            api_imp_tensor = torch.tensor(importance, dtype=torch.float32).to(self.device)
        else:
            api_seq_tensor = None
            api_imp_tensor = None
            lengths = None
        
        # 处理字节序列（如果启用且提供）
        byte_seq_tensor = None
        if self.use_byte_seq and byte_sequences is not None:
            byte_seq_tensor = torch.tensor(byte_sequences, dtype=torch.float32).to(self.device)
        
        # 处理图像（如果启用且提供）
        img_tensor = None
        if self.use_image and images is not None:
            img_tensor = torch.tensor(images, dtype=torch.float32).to(self.device)
            
        # 处理标签（如果提供）
        if labels is not None:
            labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        else:
            labels_tensor = None
            
        return pe_features_tensor, api_seq_tensor, api_imp_tensor, lengths, byte_seq_tensor, img_tensor, labels_tensor
    
    def train(self, pe_features, api_sequences, labels, val_pe_features=None, 
             val_api_sequences=None, val_labels=None, byte_sequences=None, images=None,
             val_byte_sequences=None, val_images=None, num_workers=4):
        """
        训练模型
        
        Args:
            pe_features: PE特征DataFrame
            api_sequences: API序列列表
            labels: 标签数组
            val_pe_features: 验证集PE特征
            val_api_sequences: 验证集API序列
            val_labels: 验证集标签
            byte_sequences: 字节序列特征
            images: 图像特征
            val_byte_sequences: 验证集字节序列特征
            val_images: 验证集图像特征
            num_workers: 数据加载器工作进程数
        """
        # 创建训练集和验证集
        train_dataset = MalwareDataset(
            pe_features, api_sequences, labels, 
            byte_sequences, images, self.tokenizer
        )
        
        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # 使用pin_memory加速数据传输到GPU
            collate_fn=collate_fn,
            drop_last=False
        )
        
        # 创建验证集数据加载器(如果提供验证集)
        val_loader = None
        if val_pe_features is not None and val_api_sequences is not None and val_labels is not None:
            val_dataset = MalwareDataset(
                val_pe_features, val_api_sequences, val_labels,
                val_byte_sequences, val_images, self.tokenizer
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=False
            )
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        # 训练循环
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            # 使用数据加载器进行训练
            for batch in train_loader:
                # 将数据移动到设备
                pe_features_batch = batch['pe_features'].to(self.device)
                api_seqs_batch = batch['api_seqs'].to(self.device)
                seq_lens_batch = batch['seq_lens'].to(self.device)
                api_imps_batch = batch['api_imps'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                
                # 处理其他可选特征
                byte_seqs_batch = None
                if batch['byte_seqs'] is not None:
                    byte_seqs_batch = batch['byte_seqs'].to(self.device)
                    
                images_batch = None
                if batch['images'] is not None:
                    images_batch = batch['images'].to(self.device)
                
                # 前向传播与反向传播（使用混合精度训练）
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    # 使用混合精度训练
                    with amp.autocast():
                        outputs, _, _, _ = self.model(
                            pe_features_batch, api_seqs_batch, api_imps_batch, 
                            seq_lens_batch, byte_seqs_batch, images_batch
                        )
                        loss = self.criterion(outputs, labels_batch)
                    
                    # 使用scaler进行反向传播和优化
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 常规训练
                    outputs, _, _, _ = self.model(
                        pe_features_batch, api_seqs_batch, api_imps_batch, 
                        seq_lens_batch, byte_seqs_batch, images_batch
                    )
                    loss = self.criterion(outputs, labels_batch)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # 更新参数
                    self.optimizer.step()
                
                # 更新SWA模型（如果达到了SWA开始轮数）
                if epoch >= self.swa_start:
                    self.swa_model.update_parameters(self.model)
                
                epoch_loss += loss.item()
            
            # 更新SWA学习率
            if epoch >= self.swa_start:
                self.swa_scheduler.step()
            
            # 计算平均损失
            epoch_loss /= len(train_loader)
            
            # 验证
            if val_loader is not None:
                val_loss, val_auc = self.evaluate_loader(val_loader)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), 'best_model.pt')
                
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}')
                
        # 如果使用了SWA，更新BatchNorm统计量
        if self.swa_start < self.epochs:
            print("更新SWA模型的BatchNorm统计量...")
            self.swa_model.to(self.device)
            
            # 使用训练数据加载器更新BN统计量
            with torch.no_grad():
                for batch in train_loader:
                    # 将数据移动到设备
                    pe_features_batch = batch['pe_features'].to(self.device)
                    api_seqs_batch = batch['api_seqs'].to(self.device)
                    seq_lens_batch = batch['seq_lens'].to(self.device)
                    api_imps_batch = batch['api_imps'].to(self.device)
                    
                    # 处理其他可选特征
                    byte_seqs_batch = None
                    if batch['byte_seqs'] is not None:
                        byte_seqs_batch = batch['byte_seqs'].to(self.device)
                        
                    images_batch = None
                    if batch['images'] is not None:
                        images_batch = batch['images'].to(self.device)
                    
                    # 更新BN统计量
                    self.swa_model(pe_features_batch, api_seqs_batch, api_imps_batch, 
                                   seq_lens_batch, byte_seqs_batch, images_batch)
            
            # 保存SWA模型
            torch.save(self.swa_model.state_dict(), 'swa_model.pt')
            
        # 加载最佳模型
        if val_loader is not None:
            print(f"加载最佳模型 (Epoch {best_epoch+1})")
            self.model.load_state_dict(torch.load('best_model.pt'))
    
    def evaluate_loader(self, data_loader):
        """
        使用数据加载器评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            loss: 损失值
            auc: AUC值
        """
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 将数据移动到设备
                pe_features_batch = batch['pe_features'].to(self.device)
                api_seqs_batch = batch['api_seqs'].to(self.device)
                seq_lens_batch = batch['seq_lens'].to(self.device)
                api_imps_batch = batch['api_imps'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                
                # 处理其他可选特征
                byte_seqs_batch = None
                if batch['byte_seqs'] is not None:
                    byte_seqs_batch = batch['byte_seqs'].to(self.device)
                    
                images_batch = None
                if batch['images'] is not None:
                    images_batch = batch['images'].to(self.device)
                
                # 前向传播
                outputs, _, _, _ = self.model(
                    pe_features_batch, api_seqs_batch, api_imps_batch, 
                    seq_lens_batch, byte_seqs_batch, images_batch
                )
                
                # 计算损失
                loss = self.criterion(outputs, labels_batch)
                total_loss += loss.item()
                
                # 收集预测和标签
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels_batch.cpu().numpy())
                
        # 拼接预测和标签
        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_outputs)
        
        # 计算平均损失
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, auc
    
    def predict(self, pe_features, api_sequences=None, use_swa=True, byte_sequences=None, images=None):
        """
        预测样本
        
        Args:
            pe_features: PE特征DataFrame
            api_sequences: API序列列表
            use_swa: 是否使用SWA模型
            byte_sequences: 字节序列特征
            images: 图像特征
            
        Returns:
            predictions: 预测标签
            probabilities: 预测概率
        """
        model = self.swa_model if use_swa and hasattr(self, 'swa_model') else self.model
        model.eval()
        
        n_samples = len(pe_features)
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        all_probs = []
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                batch_pe = pe_features.iloc[start_idx:end_idx]
                batch_api = None if api_sequences is None else api_sequences[start_idx:end_idx]
                
                # 取出字节序列和图像批次（如果有）
                batch_bytes = None
                if byte_sequences is not None:
                    batch_bytes = byte_sequences[start_idx:end_idx]
                    
                batch_images = None
                if images is not None:
                    batch_images = images[start_idx:end_idx]
                
                # 准备批次数据
                pe_tensor, api_tensor, imp_tensor, lengths, byte_tensor, img_tensor, _ = self.prepare_batch(
                    batch_pe, batch_api, None, batch_bytes, batch_images
                )
                
                # 前向传播
                outputs, _, _, _ = model(
                    pe_tensor, api_tensor, imp_tensor, lengths, byte_tensor, img_tensor
                )
                
                # 收集预测概率
                all_probs.append(outputs.cpu().numpy())
                
        # 拼接预测概率
        all_probs = np.vstack(all_probs).flatten()
        
        # 阈值为0.5的预测
        predictions = (all_probs >= 0.5).astype(int)
        
        return predictions, all_probs
    
    def save_model(self, model_path, swa_model_path=None):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
            swa_model_path: SWA模型保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'pe_feature_dim': self.model.pe_feature_norm.num_features,
                'vocab_size': self.model.bilstm_attention.embedding.num_embeddings,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'lstm_layers': self.lstm_layers,
                'fc_hidden': self.fc_hidden,
                'dropout': self.dropout,
                'use_byte_seq': self.use_byte_seq,
                'use_image': self.use_image
            },
            'tokenizer': self.tokenizer
        }, model_path)
        
        if swa_model_path and hasattr(self, 'swa_model'):
            torch.save({
                'swa_model_state_dict': self.swa_model.state_dict()
            }, swa_model_path)
            
    @classmethod
    def load_model(cls, model_path, swa_model_path=None, device=None):
        """
        加载模型
        
        Args:
            model_path: 模型路径
            swa_model_path: SWA模型路径
            device: 运行设备
            
        Returns:
            detector: 加载的检测器实例
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 加载模型数据
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取模型配置
        config = checkpoint['model_config']
        
        # 创建检测器实例
        detector = cls(
            pe_feature_dim=config['pe_feature_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            lstm_layers=config['lstm_layers'],
            fc_hidden=config['fc_hidden'],
            dropout=config['dropout'],
            use_byte_seq=config.get('use_byte_seq', True),
            use_image=config.get('use_image', True)
        )
        
        # 载入模型参数
        detector.model.load_state_dict(checkpoint['model_state_dict'])
        detector.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        detector.tokenizer = checkpoint['tokenizer']
        
        # 载入SWA模型参数
        if swa_model_path:
            swa_checkpoint = torch.load(swa_model_path, map_location=device)
            detector.swa_model.load_state_dict(swa_checkpoint['swa_model_state_dict'])
            
        return detector

class EnsembleModel:
    """
    集成模型 - 结合深度学习模型和LightGBM模型
    """
    def __init__(self, deep_model=None, lgbm_model=None, ensemble_weights=None):
        """
        初始化集成模型
        
        Args:
            deep_model: 深度学习模型(DeepMalwareDetector实例)
            lgbm_model: LightGBM模型(MalwareDetector实例)
            ensemble_weights: 模型权重[deep_weight, lgbm_weight]
        """
        self.deep_model = deep_model
        self.lgbm_model = lgbm_model
        self.ensemble_weights = ensemble_weights or [0.6, 0.4]  # 默认权重
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def predict(self, pe_features, api_sequences=None, byte_sequences=None, images=None, 
                use_deep_swa=True, return_separate_probs=False):
        """
        预测样本
        
        Args:
            pe_features: PE特征DataFrame
            api_sequences: API序列列表
            byte_sequences: 字节序列特征
            images: 图像特征
            use_deep_swa: 是否使用SWA深度学习模型
            return_separate_probs: 是否返回各模型的单独预测结果
            
        Returns:
            predictions: 预测标签
            probabilities: 预测概率
            (可选) model_probs: 各模型的预测概率
        """
        deep_probs = None
        lgbm_probs = None
        
        # 深度学习模型预测
        if self.deep_model is not None:
            _, deep_probs = self.deep_model.predict(
                pe_features=pe_features,
                api_sequences=api_sequences,
                use_swa=use_deep_swa,
                byte_sequences=byte_sequences,
                images=images
            )
        
        # LightGBM模型预测
        if self.lgbm_model is not None:
            # 注意：LightGBM可能期望对特征进行不同的预处理
            _, lgbm_probs = self.lgbm_model.predict(pe_features)
        
        # 集成预测
        ensemble_probs = None
        
        if deep_probs is not None and lgbm_probs is not None:
            # 加权融合
            deep_weight, lgbm_weight = self.ensemble_weights
            ensemble_probs = deep_weight * deep_probs + lgbm_weight * lgbm_probs
        elif deep_probs is not None:
            ensemble_probs = deep_probs
        elif lgbm_probs is not None:
            ensemble_probs = lgbm_probs
        else:
            raise ValueError("至少需要一个可用的模型")
        
        # 二分类阈值
        predictions = (ensemble_probs >= 0.5).astype(int)
        
        if return_separate_probs:
            return predictions, ensemble_probs, {'deep': deep_probs, 'lgbm': lgbm_probs}
        
        return predictions, ensemble_probs
    
    def save(self, ensemble_path, deep_model_path=None, swa_model_path=None, lgbm_model_path=None):
        """
        保存集成模型
        
        Args:
            ensemble_path: 集成模型保存路径
            deep_model_path: 深度学习模型保存路径
            swa_model_path: SWA深度学习模型保存路径
            lgbm_model_path: LightGBM模型保存路径
        """
        # 保存深度学习模型
        if self.deep_model is not None and deep_model_path:
            self.deep_model.save_model(deep_model_path, swa_model_path)
        
        # 保存LightGBM模型
        if self.lgbm_model is not None and lgbm_model_path:
            self.lgbm_model.save_model(lgbm_model_path)
        
        # 保存集成配置
        ensemble_config = {
            'ensemble_weights': self.ensemble_weights,
            'deep_model_path': deep_model_path,
            'swa_model_path': swa_model_path,
            'lgbm_model_path': lgbm_model_path
        }
        
        joblib.dump(ensemble_config, ensemble_path)
        
    @classmethod
    def load(cls, ensemble_path, deep_model_path=None, swa_model_path=None, 
             lgbm_model_path=None, deep_feature_engineer_path=None, lgbm_feature_engineer_path=None):
        """
        加载集成模型
        
        Args:
            ensemble_path: 集成模型路径
            deep_model_path: 深度学习模型路径(可选，覆盖ensemble_path中的配置)
            swa_model_path: SWA深度学习模型路径(可选，覆盖ensemble_path中的配置)
            lgbm_model_path: LightGBM模型路径(可选，覆盖ensemble_path中的配置)
            deep_feature_engineer_path: 深度学习特征工程器路径
            lgbm_feature_engineer_path: LightGBM特征工程器路径
            
        Returns:
            ensemble_model: 加载的集成模型
        """
        # 加载集成配置
        ensemble_config = joblib.load(ensemble_path)
        
        # 获取模型路径(优先使用参数指定的路径)
        deep_model_path = deep_model_path or ensemble_config.get('deep_model_path')
        swa_model_path = swa_model_path or ensemble_config.get('swa_model_path')
        lgbm_model_path = lgbm_model_path or ensemble_config.get('lgbm_model_path')
        
        # 加载深度学习模型
        deep_model = None
        if deep_model_path:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            deep_model = DeepMalwareDetector.load_model(deep_model_path, swa_model_path, device)
        
        # 加载LightGBM模型
        lgbm_model = None
        if lgbm_model_path:
            lgbm_model = LGBMDetector.load_model(lgbm_model_path)
        
        # 创建集成模型
        ensemble_model = cls(
            deep_model=deep_model,
            lgbm_model=lgbm_model,
            ensemble_weights=ensemble_config['ensemble_weights']
        )
        
        return ensemble_model 