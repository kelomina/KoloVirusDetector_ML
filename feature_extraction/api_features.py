import pefile
import numpy as np
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# 危险API列表（按类别分组）
DANGEROUS_APIS = {
    'process_manipulation': [
        'CreateProcess', 'CreateProcessA', 'CreateProcessW',
        'CreateRemoteThread', 'CreateRemoteThreadEx',
        'OpenProcess', 'TerminateProcess', 'ExitProcess',
        'NtCreateProcess', 'NtCreateProcessEx', 'NtOpenProcess',
        'ZwCreateProcess', 'ZwCreateProcessEx', 'ZwOpenProcess',
    ],
    'memory_manipulation': [
        'VirtualAlloc', 'VirtualAllocEx', 'VirtualProtect', 'VirtualProtectEx',
        'WriteProcessMemory', 'ReadProcessMemory',
        'NtAllocateVirtualMemory', 'NtWriteVirtualMemory', 'NtReadVirtualMemory',
        'ZwAllocateVirtualMemory', 'ZwWriteVirtualMemory', 'ZwReadVirtualMemory',
        'HeapCreate', 'HeapAlloc',
    ],
    'code_injection': [
        'CreateRemoteThread', 'CreateRemoteThreadEx', 'QueueUserAPC',
        'NtQueueApcThread', 'ZwQueueApcThread', 'RtlCreateUserThread',
    ],
    'dll_loading': [
        'LoadLibrary', 'LoadLibraryA', 'LoadLibraryW', 'LoadLibraryEx', 'LoadLibraryExA', 'LoadLibraryExW',
        'GetProcAddress', 'GetModuleHandle', 'GetModuleHandleA', 'GetModuleHandleW',
        'LdrLoadDll', 'LdrGetProcedureAddress',
    ],
    'registry_manipulation': [
        'RegOpenKey', 'RegOpenKeyA', 'RegOpenKeyW', 'RegOpenKeyEx', 'RegOpenKeyExA', 'RegOpenKeyExW',
        'RegCreateKey', 'RegCreateKeyA', 'RegCreateKeyW', 'RegCreateKeyEx', 'RegCreateKeyExA', 'RegCreateKeyExW',
        'RegDeleteKey', 'RegDeleteKeyA', 'RegDeleteKeyW', 'RegSetValue', 'RegSetValueA', 'RegSetValueW',
        'RegSetValueEx', 'RegSetValueExA', 'RegSetValueExW',
    ],
    'file_operations': [
        'CreateFile', 'CreateFileA', 'CreateFileW', 'WriteFile', 'DeleteFile', 'DeleteFileA', 'DeleteFileW',
        'CopyFile', 'CopyFileA', 'CopyFileW', 'MoveFile', 'MoveFileA', 'MoveFileW',
        'NtCreateFile', 'ZwCreateFile', 'NtOpenFile', 'ZwOpenFile',
    ],
    'network': [
        'socket', 'connect', 'bind', 'listen', 'accept', 'send', 'recv', 'WSAStartup', 'WSASocket',
        'HttpOpenRequest', 'HttpSendRequest', 'InternetOpen', 'InternetOpenUrl', 'InternetConnect',
        'InternetReadFile', 'InternetWriteFile',
    ],
    'system_info': [
        'GetSystemDirectory', 'GetWindowsDirectory', 'GetComputerName', 'GetComputerNameA', 'GetComputerNameW',
        'GetSystemInfo', 'IsDebuggerPresent', 'CheckRemoteDebuggerPresent',
    ],
    'crypto': [
        'CryptAcquireContext', 'CryptCreateHash', 'CryptHashData', 'CryptEncrypt', 'CryptDecrypt',
        'CryptGenKey', 'CryptExportKey', 'CryptImportKey',
    ],
    'anti_debug': [
        'IsDebuggerPresent', 'CheckRemoteDebuggerPresent', 'OutputDebugString', 'OutputDebugStringA', 'OutputDebugStringW',
        'NtQueryInformationProcess', 'ZwQueryInformationProcess',
    ],
}

# 字符串模式
SUSPICIOUS_PATTERNS = [
    # URLs和IP地址
    r'https?://[^\s/$.?#].[^\s]*',
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    # 可执行文件和脚本
    r'\.exe\b', r'\.dll\b', r'\.bat\b', r'\.cmd\b', r'\.ps1\b', r'\.vbs\b', r'\.js\b',
    # 注册表路径
    r'HKEY_[A-Z_]+', r'SOFTWARE\\', r'SYSTEM\\',
    # 命令行工具和参数
    r'cmd\.exe', r'powershell\.exe', r'-exec\s+bypass',
    # 加密相关
    r'aes', r'rsa', r'md5', r'sha1', r'sha256', r'encrypt', r'decrypt',
    # Windows API
    r'kernel32', r'user32', r'advapi32', r'CreateProcess', r'VirtualAlloc',
    # C2服务器相关
    r'beacon', r'callback', r'command', r'control', r'server', r'backdoor', r'remote',
    # 混淆和编码
    r'base64', r'encode', r'decode', r'obfuscate', r'xor',
]

def extract_api_features(file_path):
    """
    提取PE文件中的API调用特征
    
    Args:
        file_path: PE文件路径
    
    Returns:
        dict: API调用特征字典
    """
    try:
        pe = pefile.PE(file_path)
        features = {}
        
        # 初始化API计数器
        api_category_counts = {category: 0 for category in DANGEROUS_APIS}
        specific_api_counts = {}
        
        # 获取所有导入的DLL和API
        imported_apis = []
        api_sequence = []  # 用于API调用序列
        
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('ascii', 'ignore').lower()
                
                for imp in entry.imports:
                    if imp.name:
                        api_name = imp.name.decode('ascii', 'ignore')
                        full_api = f"{dll_name}:{api_name}"
                        imported_apis.append(api_name)
                        api_sequence.append(api_name)
                        
                        # 检查危险API
                        for category, apis in DANGEROUS_APIS.items():
                            if api_name in apis:
                                api_category_counts[category] += 1
                                specific_api_counts[api_name] = specific_api_counts.get(api_name, 0) + 1
        
        # 1. API类别特征
        for category, count in api_category_counts.items():
            features[f'api_category_{category}'] = count
            # 类别比例
            features[f'api_category_{category}_ratio'] = count / len(imported_apis) if imported_apis else 0
        
        # 2. 常见危险API的直接特征
        flat_dangerous_apis = [api for apis in DANGEROUS_APIS.values() for api in apis]
        top_apis = flat_dangerous_apis[:50]  # 选取前50个重要API
        for api in top_apis:
            features[f'api_{api}'] = 1 if api in specific_api_counts else 0
        
        # 3. API调用统计特征
        features['api_count_total'] = len(imported_apis)
        features['api_count_unique'] = len(set(imported_apis))
        features['api_dangerous_ratio'] = sum(specific_api_counts.values()) / len(imported_apis) if imported_apis else 0
        
        # 4. 滑动窗口熵编码 - 捕捉API序列模式
        if len(api_sequence) >= 5:
            window_size = 5
            entropy_values = []
            
            for i in range(len(api_sequence) - window_size + 1):
                window = api_sequence[i:i+window_size]
                counter = Counter(window)
                # 计算窗口内API调用的熵
                probs = [count/window_size for count in counter.values()]
                entropy = -sum(p * np.log2(p) for p in probs)
                entropy_values.append(entropy)
            
            if entropy_values:
                features['api_seq_entropy_mean'] = np.mean(entropy_values)
                features['api_seq_entropy_std'] = np.std(entropy_values)
                features['api_seq_entropy_max'] = np.max(entropy_values)
                features['api_seq_entropy_min'] = np.min(entropy_values)
        else:
            features['api_seq_entropy_mean'] = 0
            features['api_seq_entropy_std'] = 0
            features['api_seq_entropy_max'] = 0
            features['api_seq_entropy_min'] = 0
        
        # 5. API拓扑特征 - API调用关系
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            dll_count = len(pe.DIRECTORY_ENTRY_IMPORT)
            features['dll_count'] = dll_count
            
            # DLL和API的比例
            features['api_per_dll_ratio'] = len(imported_apis) / dll_count if dll_count > 0 else 0
            
            # 危险DLL数量
            dangerous_dlls = ['kernel32.dll', 'advapi32.dll', 'user32.dll', 'ntdll.dll', 'ws2_32.dll', 'wininet.dll']
            features['dangerous_dll_count'] = sum(1 for entry in pe.DIRECTORY_ENTRY_IMPORT 
                                                if entry.dll.decode('ascii', 'ignore').lower() in dangerous_dlls)
        else:
            features['dll_count'] = 0
            features['api_per_dll_ratio'] = 0
            features['dangerous_dll_count'] = 0
        
        return features
    
    except Exception as e:
        print(f"提取API特征时出错 {file_path}: {str(e)}")
        return {}

def extract_string_features(file_path, min_length=5, max_strings=10000):
    """
    提取PE文件中的字符串特征
    
    Args:
        file_path: PE文件路径
        min_length: 最小字符串长度
        max_strings: 最大字符串数量
        
    Returns:
        dict: 字符串特征字典
    """
    try:
        pe = pefile.PE(file_path)
        features = {}
        
        # 提取所有可打印ASCII字符串
        pe_data = pe.__data__
        ascii_strings = []
        unicode_strings = []
        
        # ASCII字符串提取
        ascii_pattern = re.compile(b'[\x20-\x7E]{%d,}' % min_length)
        ascii_strings = [s.decode('ascii') for s in ascii_pattern.findall(pe_data)][:max_strings]
        
        # Unicode字符串提取 (简化处理)
        for i in range(0, len(pe_data)-min_length*2, 2):
            if all(pe_data[j] >= 0x20 and pe_data[j] <= 0x7E and pe_data[j+1] == 0 
                   for j in range(i, i+min_length*2, 2)):
                try:
                    # 提取潜在的Unicode字符串
                    s = pe_data[i:i+min_length*8:2]
                    end = s.find(b'\x00\x00')
                    if end != -1:
                        s = s[:end]
                    unicode_strings.append(s.decode('ascii'))
                except Exception:
                    pass
        
        # 合并所有字符串
        all_strings = ascii_strings + unicode_strings
        
        if all_strings:
            # 基本字符串统计
            features['string_count'] = len(all_strings)
            features['string_avg_length'] = np.mean([len(s) for s in all_strings])
            
            # 特殊字符串模式匹配
            suspicious_patterns = {pattern: 0 for pattern in SUSPICIOUS_PATTERNS}
            
            for string in all_strings:
                for pattern in SUSPICIOUS_PATTERNS:
                    if re.search(pattern, string, re.IGNORECASE):
                        suspicious_patterns[pattern] += 1
            
            # 添加模式匹配结果到特征
            for pattern, count in suspicious_patterns.items():
                # 简化特征名
                pattern_name = pattern.replace('\\', '').replace('.', '').replace('?', '')[:15]
                features[f'str_pattern_{pattern_name}'] = count
                features[f'str_pattern_{pattern_name}_ratio'] = count / len(all_strings)
            
            # 域名和URL计数
            url_pattern = re.compile(r'https?://[^\s/$.?#].[^\s]*')
            ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
            
            features['url_count'] = sum(1 for s in all_strings if url_pattern.search(s))
            features['ip_count'] = sum(1 for s in all_strings if ip_pattern.search(s))
            
            # 可执行文件和脚本引用计数
            exe_pattern = re.compile(r'\.(exe|dll|bat|cmd|ps1|vbs|js)\b', re.IGNORECASE)
            features['executable_ref_count'] = sum(1 for s in all_strings if exe_pattern.search(s))
            
            # 注册表引用计数
            reg_pattern = re.compile(r'(HKEY_|HKLM|HKCU|SOFTWARE\\|SYSTEM\\)', re.IGNORECASE)
            features['registry_ref_count'] = sum(1 for s in all_strings if reg_pattern.search(s))
            
            # 加密相关引用计数
            crypto_pattern = re.compile(r'(aes|rsa|md5|sha|encrypt|decrypt|base64)', re.IGNORECASE)
            features['crypto_ref_count'] = sum(1 for s in all_strings if crypto_pattern.search(s))
        
        else:
            features['string_count'] = 0
            features['string_avg_length'] = 0
            
            for pattern in SUSPICIOUS_PATTERNS:
                pattern_name = pattern.replace('\\', '').replace('.', '').replace('?', '')[:15]
                features[f'str_pattern_{pattern_name}'] = 0
                features[f'str_pattern_{pattern_name}_ratio'] = 0
            
            features['url_count'] = 0
            features['ip_count'] = 0
            features['executable_ref_count'] = 0
            features['registry_ref_count'] = 0
            features['crypto_ref_count'] = 0
        
        return features
    
    except Exception as e:
        print(f"提取字符串特征时出错 {file_path}: {str(e)}")
        return {} 