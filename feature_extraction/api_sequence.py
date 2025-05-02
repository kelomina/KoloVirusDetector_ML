import pefile
import numpy as np
from collections import defaultdict

# 危险API列表，用于注意力权重的计算
DANGEROUS_APIS = {
    'process_manipulation': [
        'CreateProcess', 'CreateProcessA', 'CreateProcessW',
        'CreateRemoteThread', 'CreateRemoteThreadEx',
        'OpenProcess', 'TerminateProcess', 'ExitProcess',
    ],
    'memory_manipulation': [
        'VirtualAlloc', 'VirtualAllocEx', 'VirtualProtect', 'VirtualProtectEx',
        'WriteProcessMemory', 'ReadProcessMemory',
        'HeapCreate', 'HeapAlloc',
    ],
    'code_injection': [
        'CreateRemoteThread', 'CreateRemoteThreadEx', 'QueueUserAPC',
        'RtlCreateUserThread',
    ],
    'dll_loading': [
        'LoadLibrary', 'LoadLibraryA', 'LoadLibraryW', 'LoadLibraryEx',
        'GetProcAddress', 'GetModuleHandle', 'GetModuleHandleA', 'GetModuleHandleW',
    ],
    'registry_manipulation': [
        'RegOpenKey', 'RegOpenKeyA', 'RegOpenKeyW', 'RegOpenKeyEx',
        'RegCreateKey', 'RegCreateKeyA', 'RegCreateKeyW', 'RegCreateKeyEx',
        'RegDeleteKey', 'RegDeleteKeyA', 'RegDeleteKeyW', 'RegSetValue',
    ],
    'file_operations': [
        'CreateFile', 'CreateFileA', 'CreateFileW', 'WriteFile', 
        'DeleteFile', 'DeleteFileA', 'DeleteFileW',
        'CopyFile', 'CopyFileA', 'CopyFileW', 'MoveFile',
    ],
    'network': [
        'socket', 'connect', 'bind', 'listen', 'accept', 'send', 'recv', 
        'WSAStartup', 'WSASocket', 'HttpOpenRequest', 'HttpSendRequest',
    ],
    'crypto': [
        'CryptAcquireContext', 'CryptCreateHash', 'CryptHashData', 
        'CryptEncrypt', 'CryptDecrypt', 'CryptGenKey',
    ],
    'anti_debug': [
        'IsDebuggerPresent', 'CheckRemoteDebuggerPresent', 'OutputDebugString',
        'NtQueryInformationProcess', 'ZwQueryInformationProcess',
    ],
}

# 为不同类别的危险API设置不同的权重
CATEGORY_WEIGHTS = {
    'process_manipulation': 5.0,
    'memory_manipulation': 4.5,
    'code_injection': 5.0,
    'dll_loading': 3.0,
    'registry_manipulation': 3.5,
    'file_operations': 3.0,
    'network': 4.0,
    'crypto': 3.5,
    'anti_debug': 4.0,
}

def extract_api_sequence(file_path, include_dll=False):
    """
    从PE文件中提取API调用序列
    
    Args:
        file_path: PE文件路径
        include_dll: 是否在API名称前添加DLL名称
        
    Returns:
        list: API调用序列
    """
    try:
        pe = pefile.PE(file_path)
        api_sequence = []
        
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('ascii', 'ignore').lower()
                
                for imp in entry.imports:
                    if imp.name:
                        api_name = imp.name.decode('ascii', 'ignore')
                        if include_dll:
                            full_api = f"{dll_name}:{api_name}"
                            api_sequence.append(full_api)
                        else:
                            api_sequence.append(api_name)
        
        return api_sequence
    
    except Exception as e:
        print(f"提取API序列时出错 {file_path}: {str(e)}")
        return []

def calculate_api_importance_scores():
    """
    计算各API的重要性得分，用于注意力机制
    
    Returns:
        dict: API重要性得分字典 {api_name: score}
    """
    api_importance = {}
    
    # 为所有危险API分配权重
    for category, apis in DANGEROUS_APIS.items():
        category_weight = CATEGORY_WEIGHTS.get(category, 1.0)
        for api in apis:
            api_importance[api] = category_weight
            
            # 为API名称的各种变体也分配权重（如A/W后缀版本）
            if not api.endswith(('A', 'W')):
                api_importance[api + 'A'] = category_weight
                api_importance[api + 'W'] = category_weight
            elif api.endswith('A'):
                base_api = api[:-1]
                api_importance[base_api] = category_weight
                api_importance[base_api + 'W'] = category_weight
            elif api.endswith('W'):
                base_api = api[:-1]
                api_importance[base_api] = category_weight
                api_importance[base_api + 'A'] = category_weight
    
    # 为Nt/Zw API前缀对添加权重（Windows本地API）
    nt_zw_pairs = []
    for category, apis in DANGEROUS_APIS.items():
        for api in apis:
            if api.startswith('Nt'):
                zw_api = 'Zw' + api[2:]
                nt_zw_pairs.append((api, zw_api))
            elif api.startswith('Zw'):
                nt_api = 'Nt' + api[2:]
                nt_zw_pairs.append((nt_api, api))
    
    for nt_api, zw_api in nt_zw_pairs:
        if nt_api in api_importance:
            api_importance[zw_api] = api_importance[nt_api]
        elif zw_api in api_importance:
            api_importance[nt_api] = api_importance[zw_api]
    
    # 为一般API设置默认权重
    default_weight = 1.0
    
    return api_importance, default_weight

def get_api_sequences_from_files(file_paths, include_dll=False):
    """
    从多个PE文件中提取API序列
    
    Args:
        file_paths: PE文件路径列表
        include_dll: 是否在API名称前添加DLL名称
        
    Returns:
        list: API序列列表
    """
    all_sequences = []
    
    for file_path in file_paths:
        sequence = extract_api_sequence(file_path, include_dll)
        all_sequences.append(sequence)
    
    return all_sequences 