import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations

class FeatureEngineer:
    """
    特征工程类，包含特征选择、交叉和变换等方法
    """
    def __init__(self, top_features=500):
        """
        初始化特征工程器
        
        Args:
            top_features: 互信息过滤后保留的特征数量
        """
        self.top_features = top_features
        self.selected_features = None
        self.scaler = StandardScaler()
        self.mi_scores = None
    
    def feature_cross(self, X, cross_columns=None):
        """
        特征交叉组合
        
        Args:
            X: 特征DataFrame
            cross_columns: 需要交叉的列名列表
        
        Returns:
            pd.DataFrame: 添加交叉特征后的DataFrame
        """
        if cross_columns is None:
            # 默认选择PE头特征与API特征进行交叉
            pe_header_cols = [col for col in X.columns if col.startswith(('TimeDateStamp', 'EntryPoint', 'Size'))]
            api_cols = [col for col in X.columns if col.startswith('api_')]
            
            # 限制交叉特征数量
            if len(pe_header_cols) > 5:
                pe_header_cols = pe_header_cols[:5]
            if len(api_cols) > 5:
                api_cols = api_cols[:5]
                
            cross_columns = pe_header_cols + api_cols
        
        result = X.copy()
        
        # 两两特征交叉
        for col1, col2 in combinations(cross_columns, 2):
            # 乘积特征
            result[f'cross_mul_{col1}_{col2}'] = X[col1] * X[col2]
            
            # 比率特征（避免除零）
            result[f'cross_ratio_{col1}_{col2}'] = X[col1] / (X[col2] + 1e-10)
            
            # 加和特征
            result[f'cross_sum_{col1}_{col2}'] = X[col1] + X[col2]
            
            # 差值特征
            result[f'cross_diff_{col1}_{col2}'] = X[col1] - X[col2]
        
        # EntryPoint所在节区+前5个API交叉特征
        if 'EntryPointSection' in X.columns:
            api_cols = [col for col in X.columns if col.startswith('api_')][:5]
            for api_col in api_cols:
                # 将节区名称转换为数值索引
                section_mapping = {'.text': 1, '.data': 2, '.rdata': 3, '.rsrc': 4, '.reloc': 5, 'None': 0}
                section_values = X['EntryPointSection'].map(lambda x: section_mapping.get(x, 0))
                
                # 创建交叉特征
                result[f'cross_ep_section_{api_col}'] = section_values * X[api_col]
        
        return result
    
    def mutual_info_filter(self, X, y, top_n=None):
        """
        互信息特征过滤
        
        Args:
            X: 特征矩阵
            y: 目标变量
            top_n: 保留的特征数量，默认使用初始化时设置的值
            
        Returns:
            pd.DataFrame: 过滤后的特征矩阵
        """
        if top_n is None:
            top_n = self.top_features
        
        # 计算每个特征与目标变量的互信息值
        self.mi_scores = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': self.mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        
        # 选择前N个特征
        self.selected_features = mi_df['feature'].values[:top_n]
        
        return X[self.selected_features]
    
    def adversarial_augmentation(self, X, y, n_samples=100, noise_level=0.05):
        """
        对抗样本增强
        
        Args:
            X: 特征矩阵
            y: 目标变量
            n_samples: 生成的样本数量
            noise_level: 噪声水平
            
        Returns:
            tuple: (增强后的特征矩阵, 增强后的目标变量)
        """
        X_aug = X.copy()
        y_aug = y.copy()
        
        # 只对恶意样本进行增强
        malware_idx = np.where(y == 1)[0]
        
        if len(malware_idx) == 0:
            return X_aug, y_aug
        
        # 随机选择恶意样本进行增强
        selected_idx = np.random.choice(malware_idx, size=min(n_samples, len(malware_idx)), replace=True)
        
        for idx in selected_idx:
            sample = X.iloc[idx].copy()
            
            # 对PE头特征添加随机扰动
            pe_header_cols = [col for col in X.columns if not col.startswith(('api_', 'section_', 'entropy_', 'str_'))]
            for col in pe_header_cols:
                # 添加随机噪声
                noise = np.random.normal(0, noise_level * abs(sample[col]) if sample[col] != 0 else noise_level)
                sample[col] += noise
            
            # 将增强样本添加到数据集
            X_aug = pd.concat([X_aug, pd.DataFrame([sample])], ignore_index=True)
            y_aug = np.append(y_aug, 1)  # 标记为恶意样本
        
        return X_aug, y_aug
    
    def fit_transform(self, X, y=None):
        """
        拟合并转换特征矩阵
        
        Args:
            X: 特征矩阵
            y: 目标变量，用于互信息特征选择
            
        Returns:
            pd.DataFrame: 转换后的特征矩阵
        """
        # 1. 特征标准化
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # 2. 特征交叉
        X_crossed = self.feature_cross(X_scaled)
        
        # 3. 如果有目标变量，进行互信息特征过滤
        if y is not None:
            X_filtered = self.mutual_info_filter(X_crossed, y)
            
            # 4. 对抗样本增强
            X_aug, y_aug = self.adversarial_augmentation(X_filtered, y)
            
            return X_aug, y_aug
        
        return X_crossed
    
    def transform(self, X):
        """
        转换新的特征矩阵
        
        Args:
            X: 特征矩阵
            
        Returns:
            pd.DataFrame: 转换后的特征矩阵
        """
        # 1. 特征标准化
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        
        # 2. 特征交叉
        X_crossed = self.feature_cross(X_scaled)
        
        # 3. 如果已选择特征，则进行过滤
        if self.selected_features is not None:
            # 可能存在新数据没有的特征，处理这种情况
            existing_features = [f for f in self.selected_features if f in X_crossed.columns]
            return X_crossed[existing_features]
        
        return X_crossed 