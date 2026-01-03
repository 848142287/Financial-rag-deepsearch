"""
文件编码检测工具
"""

from typing import Optional


def detect_encoding(file_path: str) -> Optional[str]:
    """检测文件编码"""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10KB
            result = chardet.detect(raw_data)
            return result.get('encoding')
    except ImportError:
        # chardet未安装，返回None
        return None
    except Exception as e:
        print(f"编码检测失败: {e}")
        return None
