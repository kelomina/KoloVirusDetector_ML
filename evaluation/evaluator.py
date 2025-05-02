import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, 
    average_precision_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ModelEvaluator:
    """
    模型评估工具，用于评估恶意软件检测模型性能
    """
    def __init__(self, model, X_test, y_test):
        """
        初始化评估器
        
        Args:
            model: 已训练的模型
            X_test: 测试集特征
            y_test: 测试集标签
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        # 获取预测结果
        self.y_pred, self.y_pred_proba = model.predict(X_test)
        
        # 计算评估指标
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self):
        """
        计算评估指标
        
        Returns:
            dict: 评估指标字典
        """
        metrics = {}
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_pred_proba)
        
        # PR曲线
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        metrics['precision_curve'] = precision
        metrics['recall_curve'] = recall
        metrics['average_precision'] = average_precision_score(self.y_test, self.y_pred_proba)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(self.y_test, self.y_pred)
        
        # 分类报告
        report = classification_report(self.y_test, self.y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # 基本指标
        metrics['accuracy'] = report['accuracy']
        metrics['precision'] = report['1']['precision']  # 恶意软件检测精度
        metrics['recall'] = report['1']['recall']        # 恶意软件检测召回率
        metrics['f1'] = report['1']['f1-score']          # 恶意软件检测F1分数
        
        return metrics
    
    def plot_roc_curve(self, save_path=None):
        """
        绘制ROC曲线
        
        Args:
            save_path: 图表保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.metrics['fpr'], self.metrics['tpr'], 
                 color='blue', lw=2, 
                 label=f'ROC曲线 (AUC = {self.metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('受试者工作特征曲线 (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        绘制精确率-召回率曲线
        
        Args:
            save_path: 图表保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.metrics['recall_curve'], self.metrics['precision_curve'], 
                 color='green', lw=2, 
                 label=f'PR曲线 (AP = {self.metrics["average_precision"]:.4f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            save_path: 图表保存路径
        """
        plt.figure(figsize=(10, 8))
        cm = self.metrics['confusion_matrix']
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建带有百分比的标注
        annot = np.zeros_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                    xticklabels=['良性', '恶意'], 
                    yticklabels=['良性', '恶意'])
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title('混淆矩阵')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, output_dir=None, report_name=None):
        """
        生成综合评估报告
        
        Args:
            output_dir: 输出目录
            report_name: 报告名称
            
        Returns:
            dict: 评估指标
        """
        if output_dir is None:
            output_dir = 'evaluation_reports'
        
        os.makedirs(output_dir, exist_ok=True)
        
        if report_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f'malware_detection_report_{timestamp}'
        
        # 生成图表
        self.plot_roc_curve(save_path=os.path.join(output_dir, f'{report_name}_roc.png'))
        self.plot_precision_recall_curve(save_path=os.path.join(output_dir, f'{report_name}_pr.png'))
        self.plot_confusion_matrix(save_path=os.path.join(output_dir, f'{report_name}_cm.png'))
        
        # 生成报告文本
        report_path = os.path.join(output_dir, f'{report_name}.txt')
        with open(report_path, 'w') as f:
            f.write('=== 恶意软件检测模型评估报告 ===\n')
            f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            f.write('--- 性能指标 ---\n')
            f.write(f'准确率 (Accuracy): {self.metrics["accuracy"]:.4f}\n')
            f.write(f'精确率 (Precision): {self.metrics["precision"]:.4f}\n')
            f.write(f'召回率 (Recall): {self.metrics["recall"]:.4f}\n')
            f.write(f'F1分数 (F1-Score): {self.metrics["f1"]:.4f}\n')
            f.write(f'ROC曲线下面积 (AUC): {self.metrics["roc_auc"]:.4f}\n')
            f.write(f'PR曲线下面积 (AP): {self.metrics["average_precision"]:.4f}\n\n')
            
            f.write('--- 混淆矩阵 ---\n')
            cm = self.metrics['confusion_matrix']
            f.write(f'真负例 (TN): {cm[0, 0]}\n')
            f.write(f'假正例 (FP): {cm[0, 1]}\n')
            f.write(f'假负例 (FN): {cm[1, 0]}\n')
            f.write(f'真正例 (TP): {cm[1, 1]}\n\n')
            
            f.write('--- 分类报告 ---\n')
            report_df = pd.DataFrame(self.metrics['classification_report']).transpose()
            f.write(report_df.to_string() + '\n')
        
        print(f'评估报告已保存到: {report_path}')
        
        # 返回关键指标
        return {
            'accuracy': self.metrics['accuracy'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'f1': self.metrics['f1'],
            'roc_auc': self.metrics['roc_auc'],
            'average_precision': self.metrics['average_precision']
        } 