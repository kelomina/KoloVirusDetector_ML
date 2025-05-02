import pefile
import numpy as np
import math
from collections import defaultdict

def extract_section_features(file_path, max_sections=10):
    """
    提取PE文件的节区特征
    
    Args:
        file_path: PE文件路径
        max_sections: 最大节区数量限制
    
    Returns:
        dict: 节区特征字典
    """
    try:
        pe = pefile.PE(file_path)
        features = {}
        
        # 基本节区数量统计
        features['num_sections'] = len(pe.sections)
        
        # 常见节区名计数
        common_sections = {'.text': 0, '.data': 0, '.rdata': 0, '.rsrc': 0, '.reloc': 0, 
                           '.idata': 0, '.edata': 0, '.pdata': 0, '.bss': 0, '.tls': 0}
        
        # 节区特征计算
        section_sizes = []
        section_entropies = []
        section_characteristics = defaultdict(int)
        
        # 循环处理每个节区
        raw_sizes = []
        virtual_sizes = []
        raw_ptr_diffs = []
        entropy_mean = 0
        
        # 遍历所有节区并提取特征
        for i, section in enumerate(pe.sections):
            if i >= max_sections:
                break
                
            section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
            
            # 更新通用节区计数
            if section_name in common_sections:
                common_sections[section_name] = 1
                
            # 节区基本特征
            prefix = f'section_{i}_'
            features[prefix + 'raw_size'] = section.SizeOfRawData
            features[prefix + 'virtual_size'] = section.Misc_VirtualSize
            features[prefix + 'entropy'] = section.get_entropy()
            features[prefix + 'virt_addr'] = section.VirtualAddress
            features[prefix + 'raw_ptr'] = section.PointerToRawData
            
            # 计算虚拟大小与原始大小的比率
            if section.SizeOfRawData > 0:
                features[prefix + 'virt_raw_ratio'] = section.Misc_VirtualSize / section.SizeOfRawData
            else:
                features[prefix + 'virt_raw_ratio'] = 0
            
            # 节区特征：特征标志位
            characteristics = section.Characteristics
            features[prefix + 'characteristic_code'] = 1 if characteristics & 0x00000020 else 0
            features[prefix + 'characteristic_initialized'] = 1 if characteristics & 0x00000040 else 0
            features[prefix + 'characteristic_uninitialized'] = 1 if characteristics & 0x00000080 else 0
            features[prefix + 'characteristic_discardable'] = 1 if characteristics & 0x02000000 else 0
            features[prefix + 'characteristic_executable'] = 1 if characteristics & 0x20000000 else 0
            features[prefix + 'characteristic_readable'] = 1 if characteristics & 0x40000000 else 0
            features[prefix + 'characteristic_writable'] = 1 if characteristics & 0x80000000 else 0
            
            # 收集统计信息
            raw_sizes.append(section.SizeOfRawData)
            virtual_sizes.append(section.Misc_VirtualSize)
            if section.PointerToRawData > 0:
                raw_ptr_diffs.append(section.PointerToRawData)
            entropy_mean += section.get_entropy()
            
            # 节区内容特征
            if section.SizeOfRawData > 0 and section.PointerToRawData > 0:
                try:
                    section_data = pe.__data__[section.PointerToRawData:section.PointerToRawData+section.SizeOfRawData]
                    
                    # 零字节百分比
                    zero_count = section_data.count(0)
                    features[prefix + 'zero_ratio'] = zero_count / len(section_data) if len(section_data) > 0 else 0
                    
                    # 字节直方图
                    byte_counts = np.zeros(16)  # 16个bin（0x0-0xF）
                    for byte in section_data:
                        byte_counts[byte >> 4] += 1  # 取高4位作为bin索引
                    
                    for j in range(16):
                        features[prefix + f'byte_hist_{j}'] = byte_counts[j] / len(section_data) if len(section_data) > 0 else 0
                        
                    # ASCII字符比例
                    printable_count = sum(32 <= b <= 126 for b in section_data)
                    features[prefix + 'printable_ratio'] = printable_count / len(section_data) if len(section_data) > 0 else 0
                
                except Exception:
                    features[prefix + 'zero_ratio'] = 0
                    for j in range(16):
                        features[prefix + f'byte_hist_{j}'] = 0
                    features[prefix + 'printable_ratio'] = 0
        
        # 将通用节区存在标志添加到特征中
        for section_name, present in common_sections.items():
            features[f'has_{section_name.replace(".", "")}'] = present
            
        # 节区统计特征
        if len(pe.sections) > 0:
            entropy_mean /= len(pe.sections)
            
        features['section_entropy_mean'] = entropy_mean
        
        if raw_sizes:
            features['section_raw_size_mean'] = np.mean(raw_sizes)
            features['section_raw_size_max'] = np.max(raw_sizes)
            features['section_raw_size_min'] = np.min(raw_sizes)
            features['section_raw_size_std'] = np.std(raw_sizes) if len(raw_sizes) > 1 else 0
        
        if virtual_sizes:
            features['section_virt_size_mean'] = np.mean(virtual_sizes)
            features['section_virt_size_max'] = np.max(virtual_sizes)
            features['section_virt_size_min'] = np.min(virtual_sizes)
            features['section_virt_size_std'] = np.std(virtual_sizes) if len(virtual_sizes) > 1 else 0
         
        if raw_ptr_diffs and len(raw_ptr_diffs) > 1:
            # 相邻节区之间的指针差异
            diffs = [raw_ptr_diffs[i+1] - raw_ptr_diffs[i] for i in range(len(raw_ptr_diffs)-1)]
            features['section_ptr_diff_mean'] = np.mean(diffs) if diffs else 0
            features['section_ptr_diff_std'] = np.std(diffs) if len(diffs) > 1 else 0
        
        return features
    
    except Exception as e:
        print(f"提取节区特征时出错 {file_path}: {str(e)}")
        return {} 