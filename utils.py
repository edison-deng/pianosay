# -*- coding: utf-8 -*-

import sys
import os
import io
import numpy as np
import mido
import torch
from scipy.io.wavfile import write as write_wav

import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image

try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                             QLabel, QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox,
                             QFileDialog, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QRadioButton, QMessageBox, QSlider, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt

# 重定向模型缓存目录
def setup_cache():
    try:
        user_home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(user_home_dir, ".demucs_model_cache")

        if 'TORCH_HOME' not in os.environ:
            os.environ['TORCH_HOME'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            print(f"[INFO] PyTorch/Demucs模型缓存目录已设置为: {cache_dir}")
    except Exception as e:
        print(f"[ERROR] 设置缓存目录失败: {e}。将使用默认位置。")

# MIDI到Hz转换（共享函数）
def midi_to_hz(note):
    return (440.0 / 32) * (2 ** ((note - 9) / 12))