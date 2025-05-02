import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MalwareDetector:
    """
    恶意软件检测模型，基于LightGBM
    """
    def __init__(self, params=None, n_estimators=1000, early_stopping_rounds=50):
        """
        初始化检测器
        
        Args:
            params: LightGBM参数字典
            n_estimators: 最大树数量
            early_stopping_rounds: 早停轮数
        """
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        
        # 默认LightGBM参数
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }
        
        # 更新用户提供的参数
        if params:
            self.params.update(params)
            
        self.model = None
        self.feature_importance = None
    
    def train(self, X, y, X_val=None, y_val=None, test_size=0.2, random_state=42):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            test_size: 训练集分割比例
            random_state: 随机种子
            
        Returns:
            dict: 验证集上的评估指标
        """
        # 如果未提供验证集，则分割训练集
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[val_data],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100
        )
        
        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance('gain')
        }).sort_values('importance', ascending=False)
        
        # 在验证集上评估
        y_pred_proba = self.model.predict(X_val)
        
        # 确定最佳阈值
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_val, y_pred_proba >= t) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        
        # 使用最佳阈值进行预测
        y_pred = y_pred_proba >= best_threshold
        
        # 计算评估指标
        metrics = {
            'auc': roc_auc_score(y_val, y_pred_proba),
            'accuracy': np.mean(y_pred == y_val),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'threshold': best_threshold
        }
        
        # 保存阈值
        self.threshold = best_threshold
        
        # 计算并保存混淆矩阵
        self.confusion_matrix = confusion_matrix(y_val, y_pred)
        
        return metrics
    
    def predict(self, X, threshold=None):
        """
        对新样本进行预测
        
        Args:
            X: 特征矩阵
            threshold: 分类阈值，默认使用训练时确定的最佳阈值
            
        Returns:
            np.array: 预测标签
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        # 获取预测概率
        y_pred_proba = self.model.predict(X)
        
        # 使用指定阈值或训练时确定的最佳阈值
        if threshold is None:
            threshold = getattr(self, 'threshold', 0.5)
        
        # 返回二分类结果
        return y_pred_proba >= threshold, y_pred_proba
    
    def cross_validate(self, X, y, n_splits=5, random_state=42):
        """
        交叉验证
        
        Args:
            X: 特征矩阵
            y: 目标变量
            n_splits: 折数
            random_state: 随机种子
            
        Returns:
            dict: 交叉验证评估指标均值和标准差
        """
        # 初始化折叠
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # 初始化评估指标列表
        metrics_list = []
        
        # 执行交叉验证
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建新模型
            model = MalwareDetector(params=self.params, n_estimators=self.n_estimators, 
                                    early_stopping_rounds=self.early_stopping_rounds)
            
            # 训练并评估
            metrics = model.train(X_train, y_train, X_val, y_val)
            metrics_list.append(metrics)
        
        # 计算平均指标和标准差
        avg_metrics = {}
        std_metrics = {}
        
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[f'avg_{key}'] = np.mean(values)
            std_metrics[f'std_{key}'] = np.std(values)
        
        # 合并结果
        result = {**avg_metrics, **std_metrics}
        
        return result
    
    def feature_importance_analysis(self, top_n=20, plot=True):
        """
        分析特征重要性
        
        Args:
            top_n: 显示的顶部特征数量
            plot: 是否绘制图表
            
        Returns:
            pd.DataFrame: 特征重要性DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        top_features = self.feature_importance.head(top_n)
        
        if plot:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top {top_n} Feature Importance')
            plt.tight_layout()
            plt.show()
        
        return top_features
    
    def save_model(self, model_path, feature_importance_path=None):
        """
        保存模型和特征重要性
        
        Args:
            model_path: 模型保存路径
            feature_importance_path: 特征重要性保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        # 保存阈值信息
        model_data = {
            'model': self.model,
            'threshold': getattr(self, 'threshold', 0.5),
            'params': self.params,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存模型
        joblib.dump(model_data, model_path)
        
        # 保存特征重要性
        if feature_importance_path and self.feature_importance is not None:
            self.feature_importance.to_csv(feature_importance_path, index=False)
    
    @classmethod
    def load_model(cls, model_path):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            MalwareDetector: 加载的检测器实例
        """
        # 加载模型数据
        model_data = joblib.load(model_path)
        
        # 创建新实例
        detector = cls(params=model_data['params'])
        detector.model = model_data['model']
        detector.threshold = model_data['threshold']
        
        return detector
    
    def optimize_hyperparameters(self, X, y, n_trials=100, timeout=3600, n_splits=5):
        """
        使用Optuna优化超参数
        
        Args:
            X: 特征矩阵
            y: 目标变量
            n_trials: 优化试验次数
            timeout: 优化超时时间（秒）
            n_splits: 交叉验证折数
            
        Returns:
            dict: 最佳参数
        """
        def objective(trial):
            # 需要优化的参数空间
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
                'verbose': -1,
                'n_jobs': -1
            }
            
            # 交叉验证
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # 使用少量迭代以加快优化
                model = lgb.train(
                    param,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[val_data],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
                
                y_pred = model.predict(X_val)
                score = roc_auc_score(y_val, y_pred)
                scores.append(score)
            
            # 返回平均AUC
            return np.mean(scores)
        
        # 创建Optuna研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 获取最佳参数
        best_params = study.best_params
        
        # 更新模型参数
        self.params.update(best_params)
        
        return best_params 