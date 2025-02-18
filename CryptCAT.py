import ctypes
from ctypes import wintypes
import uuid
import ctypes

class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8)
    ]

# 定义WinTrustData结构体
class WinTrustData(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pPolicyCallbackData", wintypes.LPVOID),
        ("pSIPClientData", wintypes.LPVOID),
        ("dwUIChoice", wintypes.DWORD),
        ("fdwRevocationChecks", wintypes.DWORD),
        ("dwUnionChoice", wintypes.DWORD),
        ("pFile", wintypes.LPVOID),
        ("dwStateAction", wintypes.DWORD),
        ("hWVTStateData", wintypes.HANDLE),
        ("pwszURLReference", wintypes.LPCWSTR),
        ("dwProvFlags", wintypes.DWORD),
        ("dwUIContext", wintypes.DWORD),
    ]

# 定义WinTrustFileInfo结构体
class WinTrustFileInfo(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pcwszFilePath", wintypes.LPCWSTR),
        ("hFile", wintypes.HANDLE),
        ("pgKnownSubject", ctypes.c_void_p),
    ]

# 加载Wintrust.dll
wintrust = ctypes.WinDLL('Wintrust.dll')

# 定义函数参数和返回类型
CryptCATAdminAcquireContext = wintrust.CryptCATAdminAcquireContext
CryptCATAdminAcquireContext.argtypes = [
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(GUID),
    wintypes.DWORD
]
CryptCATAdminAcquireContext.restype = wintypes.BOOL

CryptCATAdminReleaseContext = wintrust.CryptCATAdminReleaseContext
CryptCATAdminReleaseContext.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD
]
CryptCATAdminReleaseContext.restype = wintypes.BOOL

CryptCATAdminCalcHashFromFileHandle = wintrust.CryptCATAdminCalcHashFromFileHandle
CryptCATAdminCalcHashFromFileHandle.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(ctypes.c_ubyte),
    wintypes.DWORD
]
CryptCATAdminCalcHashFromFileHandle.restype = wintypes.BOOL

CryptCATAdminEnumCatalogFromHash = wintrust.CryptCATAdminEnumCatalogFromHash
CryptCATAdminEnumCatalogFromHash.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_ubyte),
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE)
]
CryptCATAdminEnumCatalogFromHash.restype = wintypes.BOOL

CryptCATAdminReleaseCatalogContext = wintrust.CryptCATAdminReleaseCatalogContext
CryptCATAdminReleaseCatalogContext.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD
]
CryptCATAdminReleaseCatalogContext.restype = wintypes.BOOL

def has_catalog_signature(file_path):
    hFile = ctypes.windll.kernel32.CreateFileW(
        file_path,
        0x80000000,  # GENERIC_READ
        0x00000001,  # FILE_SHARE_READ
        None,
        3,  # OPEN_EXISTING
        0x80,  # FILE_ATTRIBUTE_NORMAL
        None
    )
    if hFile == -1:
        raise Exception("无法打开文件")

    cbHash = wintypes.DWORD(0)
    if not CryptCATAdminCalcHashFromFileHandle(hFile, ctypes.byref(cbHash), None, 0):
        raise Exception("计算哈希大小失败")

    pbHash = (ctypes.c_ubyte * cbHash.value)()
    if not CryptCATAdminCalcHashFromFileHandle(hFile, ctypes.byref(cbHash), pbHash, 0):
        raise Exception("计算哈希失败")

    hCatAdmin = wintypes.HANDLE()
    if not CryptCATAdminAcquireContext(ctypes.byref(hCatAdmin), None, 0):
        raise Exception("获取目录管理上下文失败")

    hPrevCatInfo = wintypes.HANDLE()
    has_signature = CryptCATAdminEnumCatalogFromHash(hCatAdmin, pbHash, cbHash, 0, ctypes.byref(hPrevCatInfo))
    if has_signature:
        CryptCATAdminReleaseCatalogContext(hPrevCatInfo, 0)

    CryptCATAdminReleaseContext(hCatAdmin, 0)
    ctypes.windll.kernel32.CloseHandle(hFile)
    return bool(has_signature)

# 示例用法
if __name__ == "__main__":
    file_path = "E:\\样本库\\待加入白名单\\白名单3\\conhost.exe"
    try:
        if has_catalog_signature(file_path):
            print(f"文件 {file_path} 具有目录签名")
        else:
            print(f"文件 {file_path} 没有目录签名")
    except Exception as e:
        print(f"检测失败: {str(e)}")
