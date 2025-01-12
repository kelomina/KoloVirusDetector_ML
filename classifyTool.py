# How to use:
# python classifyTool.py --model ML.pkl --directory <Dir>

import os
import numpy as np
import pefile
import joblib
from tqdm import tqdm
from collections import Counter
import math

def load_trained_model(model_path):
    return joblib.load(model_path)

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
        print(f"无法提取 PE 特征: {e}")
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
            print(f"文件 {file_path} 不是有效的 PE 文件")
            return None

        selected_features = [entropy] + byte_distribution + list(first_128_bytes) + list(last_128_bytes) + pe_features

        return selected_features
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return None

def classify_pe_files(directory, model):
    results = []
    for root, _, files in os.walk(directory):
        for file_name in tqdm(files, desc="处理文件"):
            file_path = os.path.join(root, file_name)
            try:
                pe = pefile.PE(file_path)
                features = extract_features(file_path)
                if features is not None:
                    prediction = model.predict([features])[0]
                    results.append((file_path, "恶意软件" if prediction == 1 else "良性软件"))
            except pefile.PEFormatError:
                print(f"文件 {file_path} 不是有效的 PE 文件")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="恶意软件检测分类器")
    parser.add_argument('--model', type=str, required=True, help="训练好的模型路径")
    parser.add_argument('--directory', type=str, required=True, help="要分类的文件目录")
    args = parser.parse_args()

    model_path = args.model
    directory = args.directory

    model = load_trained_model(model_path)

    results = classify_pe_files(directory, model)

    with open("classification_results.txt", "w", encoding="utf-8") as f:
        for file_path, classification in results:
            f.write(f"{file_path}: {classification}\n")

    print("分类结果已保存到 classification_results.txt")
