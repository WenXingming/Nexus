"""
    递归搜索、删除 Python 缓存文件和文件夹（__pycache__）的脚本。
    使用方法：python clean_py_cache.py
"""

import pathlib

# 删除所有 .pyc 文件（注意默认当前脚本目录是 scripts，所以需要向上一级目录查找 .pyc 文件）
for pyc_file in pathlib.Path('.').parent.rglob('*.pyc'): 
   pyc_file.unlink()

# 删除所有 __pycache__ 文件夹
for cache_dir in pathlib.Path('.').parent.rglob('__pycache__'):
   cache_dir.rmdir()