import argparse
import joblib
import numpy as np
import logging

# 导入必要的函数
from train_virus_detector import extract_combined_features, calculate_entropy, string_to_fixed_length_bytes

def load_model_and_preprocessors(model_path, scaler_path, pca_path, feature_selection_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        feature_selection = joblib.load(feature_selection_path)
        #logging.info(f"模型、scaler、pca 和 feature_selection 已从 {model_path}, {scaler_path}, {pca_path}, {feature_selection_path} 加载")
        return model, scaler, pca, feature_selection
    except Exception as e:
        logging.error(f"加载模型、scaler、pca 和 feature_selection 时发生错误: {e}")
        return None, None, None, None

def predict_file(model, scaler, pca, feature_selection, file_path):
    try:
        feature = extract_combined_features(file_path)
        if feature is None:
            logging.error(f"无法提取文件 {file_path} 的特征")
            return None

        #logging.debug(f"提取的特征数量: {len(feature)}")  # 添加日志输出特征数量

        feature = np.array(feature).reshape(1, -1)
        feature_scaled = scaler.transform(feature)
        feature_compressed = pca.transform(feature_scaled)
        feature_selected = feature_selection.transform(feature_compressed)

        prediction = model.predict(feature_selected)
        return prediction[0]
    except Exception as e:
        logging.error(f"预测文件 {file_path} 时发生错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="恶意软件检测模型推理")
    parser.add_argument('--model', type=str, required=True, help="模型文件路径")
    parser.add_argument('--scaler', type=str, required=True, help="scaler 文件路径")
    parser.add_argument('--pca', type=str, required=True, help="pca 文件路径")
    parser.add_argument('--feature_selection', type=str, required=True, help="feature_selection 文件路径")
    parser.add_argument('--file', type=str, required=True, help="待推理文件路径")
    args = parser.parse_args()

    model, scaler, pca, feature_selection = load_model_and_preprocessors(args.model, args.scaler, args.pca, args.feature_selection)
    if model is None or scaler is None or pca is None or feature_selection is None:
        logging.error("模型、scaler、pca 或 feature_selection 加载失败，无法进行推理")
        return

    prediction = predict_file(model, scaler, pca, feature_selection, args.file)
    if prediction is not None:
        if prediction == 1:
            print("1")
            #logging.info(f"文件 {args.file} 被识别为恶意软件")
        else:
            print("0")
            #logging.info(f"文件 {args.file} 被识别为良性软件")

if __name__ == "__main__":
    main()
