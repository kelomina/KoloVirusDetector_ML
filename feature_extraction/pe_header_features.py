import pefile
import numpy as np
import os
import lief
from collections import defaultdict

def extract_pe_header_features(file_path):
    """
    提取PE文件头的结构化特征
    
    Args:
        file_path: PE文件路径
    
    Returns:
        dict: PE头特征字典
    """
    try:
        pe = pefile.PE(file_path)
        binary = lief.parse(file_path)
        
        features = {}
        
        # 基本PE头信息
        features['TimeDateStamp'] = pe.FILE_HEADER.TimeDateStamp
        features['PointerToSymbolTable'] = pe.FILE_HEADER.PointerToSymbolTable
        features['NumberOfSections'] = pe.FILE_HEADER.NumberOfSections
        features['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
        features['Characteristics'] = pe.FILE_HEADER.Characteristics
        
        # 可选PE头信息
        features['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
        features['MinorLinkerVersion'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
        features['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
        features['SizeOfInitializedData'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
        features['SizeOfUninitializedData'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
        features['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        features['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode
        
        # 64位PE文件中没有BaseOfData
        try:
            features['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
        except AttributeError:
            features['BaseOfData'] = 0
            
        features['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
        features['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
        features['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
        features['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
        features['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
        features['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
        features['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
        features['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
        features['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
        features['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
        features['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
        features['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
        features['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
        features['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
        features['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
        features['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
        features['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
        features['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
        features['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
        features['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
        
        # 安全特性标志
        features['ASLR'] = 1 if binary.has_nx else 0
        features['DEP'] = 1 if binary.has_aslr else 0
        features['SEH'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x0400 else 0
        features['CFG'] = 1 if pe.OPTIONAL_HEADER.DllCharacteristics & 0x4000 else 0
        
        # EntryPoint相关特征
        for section in pe.sections:
            section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
            if section.contains_rva(pe.OPTIONAL_HEADER.AddressOfEntryPoint):
                features['EntryPointSection'] = section_name
                virt_addr = section.VirtualAddress
                raw_addr = section.PointerToRawData
                ep_raw = pe.OPTIONAL_HEADER.AddressOfEntryPoint - virt_addr + raw_addr
                features['EntryPointRawAddress'] = ep_raw
                break
        else:
            features['EntryPointSection'] = 'None'
            features['EntryPointRawAddress'] = 0
        
        # 导入表/导出表大小
        try:
            features['ImportTableSize'] = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].Size
        except (IndexError, AttributeError):
            features['ImportTableSize'] = 0
        
        try:
            features['ExportTableSize'] = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].Size
        except (IndexError, AttributeError):
            features['ExportTableSize'] = 0
        
        # 资源表大小
        try:
            features['ResourceTableSize'] = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].Size
        except (IndexError, AttributeError):
            features['ResourceTableSize'] = 0
            
        # 附加特征：文件大小比率
        features['FileAlignment_SectionAlignment_Ratio'] = pe.OPTIONAL_HEADER.FileAlignment / pe.OPTIONAL_HEADER.SectionAlignment if pe.OPTIONAL_HEADER.SectionAlignment != 0 else 0
        features['SizeOfCode_SizeOfImage_Ratio'] = pe.OPTIONAL_HEADER.SizeOfCode / pe.OPTIONAL_HEADER.SizeOfImage if pe.OPTIONAL_HEADER.SizeOfImage != 0 else 0
        
        return features
    
    except Exception as e:
        print(f"提取PE头特征时出错 {file_path}: {str(e)}")
        return {} 