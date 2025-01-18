import os
import numpy as np
import pefile
import joblib
from collections import Counter
import math
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import argparse
import lightgbm

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def load_trained_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"模型加载失败: 模型文件未找到 - {model_path}")
    except (OSError, IOError):
        print(f"模型加载失败: 文件读取或格式错误 - {model_path}")
    except Exception as e:
        print(f"模型加载失败: 未知错误 - {str(e)}")
    return None

def calculate_entropy(data):
    if not data:
        return 0
    counter = Counter(data)
    length = len(data)
    entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
    return entropy

def extract_pe_features(pe):
    try:
        dos_header = pe.DOS_HEADER
        file_header = pe.FILE_HEADER
        optional_header = pe.OPTIONAL_HEADER

        features = [
            dos_header.e_magic,
            dos_header.e_lfanew,
            file_header.Machine,
            file_header.NumberOfSections,
            optional_header.AddressOfEntryPoint,
            optional_header.ImageBase,
            optional_header.SectionAlignment,
            optional_header.FileAlignment,
            optional_header.SizeOfImage,
            optional_header.SizeOfHeaders,
            optional_header.CheckSum,
            optional_header.Subsystem,
            optional_header.SizeOfStackReserve,
            optional_header.SizeOfStackCommit,
            optional_header.SizeOfHeapReserve,
            optional_header.SizeOfHeapCommit,
            optional_header.NumberOfRvaAndSizes
        ]

        text_section = next((section for section in pe.sections if b'.text' in section.Name), None)
        text_entropy = calculate_entropy(text_section.get_data()) if text_section else 0
        features.append(text_entropy)

        data_section_size = next((section.Misc_VirtualSize for section in pe.sections if b'.data' in section.Name), 0)
        features.append(data_section_size)

        imported_functions = set()
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        imported_functions.add(imp.name.decode('utf-8', errors='ignore'))
        
        sorted_imported_functions = sorted(imported_functions)[:32]
        function_names_feature = [func.encode('utf-8')[:50] for func in sorted_imported_functions]
        function_names_feature += [b'\x00'] * (32 - len(function_names_feature))
        function_names_feature = [byte for func_name in function_names_feature for byte in func_name.ljust(50, b'\x00')]
        features.extend(function_names_feature)

        description = ''
        copyright_info = ''
        if hasattr(pe, 'FileInfo'):
            for entry in pe.FileInfo:
                if hasattr(entry, 'StringTable'):
                    for st in entry.StringTable:
                        for key, value in st.entries.items():
                            if key.lower() == 'filedescription':
                                description = value.decode('utf-8', errors='ignore')
                            elif key.lower() == 'legalcopyright':
                                copyright_info = value.decode('utf-8', errors='ignore')
        
        description_bytes = description.encode('utf-8')[:128] + b'\x00' * (128 - min(len(description), 128))
        copyright_info_bytes = copyright_info.encode('utf-8')[:128] + b'\x00' * (128 - min(len(copyright_info), 128))
        features.extend(description_bytes)
        features.extend(copyright_info_bytes)

        return features
    except Exception as e:
        return None

def extract_features(file_path):
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()

        byte_distribution = [0] * 256
        for byte in file_data:
            byte_distribution[byte] += 1
        byte_distribution = [x / len(file_data) for x in byte_distribution]

        entropy = calculate_entropy(file_data)

        first_128_bytes = file_data[:128].ljust(128, b'\x00')
        last_128_bytes = file_data[-128:].rjust(128, b'\x00')

        pe_features = []
        try:
            pe = pefile.PE(file_path)
            pe_features = extract_pe_features(pe)
        except pefile.PEFormatError:
            return None

        selected_features = [entropy] + byte_distribution + list(first_128_bytes) + list(last_128_bytes) + pe_features

        return selected_features
    except Exception as e:
        return None

def process_file(file_path, model):
    try:
        features = extract_features(file_path)
        if features is not None:
            prediction = model.predict([features])[0]
            return prediction
        else:
            return -1
    except pefile.PEFormatError:
        return -1
    except Exception as e:
        return -1

def classify_pe_files(directory, model, max_workers=None):
    results = []
    all_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            all_files.append(file_path)
    
    if not all_files:
        return results

    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        for result in executor.map(lambda file_path: process_file(file_path, model), all_files):
            if result is not None:
                results.append((file_path, result))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="恶意软件检测分类器")
    parser.add_argument('--model', type=str, required=True, help="训练好的模型路径")
    parser.add_argument('--directory', type=str, default=None, help="要分类的文件目录")
    parser.add_argument('--file', type=str, default=None, help="要分类的单个文件路径")
    parser.add_argument('--max_workers', type=int, default=None, help="最大工作线程数，默认为CPU核心数")
    args = parser.parse_args()

    model_path = args.model
    directory = args.directory
    file_path = args.file
    max_workers = args.max_workers

    if not (directory or file_path):
        print("必须提供 --directory 或 --file 参数之一")
        exit(1)

    if directory and file_path:
        print("不能同时提供 --directory 和 --file 参数")
        exit(1)

    model = load_trained_model(model_path)
    if model is None:
        print("模型加载失败，退出程序")
        exit(1)

    start_time = time.time()

    if file_path:
        classification = process_file(file_path, model)
        if classification is not None:
            print(classification)
        else:
            print(-1)
    else:
        results = classify_pe_files(directory, model, max_workers=max_workers)
        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / len(results) if results else 0

        with open("classification_results.txt", "w", encoding="utf-8") as f:
            for file_path, classification in results:
                f.write(f"{file_path}: {classification}\n")

        print(f"分类结果已保存到 classification_results.txt")
        print(f"总用时: {total_time:.2f} 秒")
        print(f"平均用时: {average_time:.2f} 秒")
