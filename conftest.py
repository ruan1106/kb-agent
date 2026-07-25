"""pytest 路径修正:确保 kb 包可导入。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
