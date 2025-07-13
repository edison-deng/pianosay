```markdown
# 🎹 钢琴唱歌工具套件 (GUI版) v7.7

一个功能强大的音频分析与合成工具，支持从音频分离人声与乐器轨道，并将其智能转为MIDI文件，同时提供MIDI文件合成WAV音频的功能。

> 🚀 支持 Demucs + PyTorch 的 GPU 加速  
> 🎛️ 双处理模式（音轨分离 / EQ 模式）  
> 🔒 智能缓存与C盘保护机制  
> 📊 支持可视化频谱图生成  
> 🎶 MIDI音符模式可选（连续 / 快照）

---

## 🧰 特性 Features

- 🎼 **输入音频转MIDI**
  - 支持MP3、WAV、FLAC
  - 自动分离人声、鼓、贝斯、其他音轨
  - 可调节各音轨增益，增强MIDI识别准确性
  - 自动生成MIDI（单轨和多轨）
  - 支持EQ频段手动增益调整

- 🌈 **生成频谱图**
  - 把各音轨频谱以透明图层叠加成合成频谱图（RGBA格式）

- 🎵 **MIDI合成WAV音频**
  - 渲染MIDI为合成钢琴音轨
  - 支持批量处理，内置合成器（基于Sine波）
  - 支持复音（Polyphony）、自定义攻击/释放时间

- ⚡ **硬件加速**
  - 使用PyTorch + Demucs进行高效音轨分离，可利用GPU（如NVIDIA CUDA）

---

## 🖥️ 运行截图（可选添加）

放置项目运行截图（如GUI界面、频谱图可视化等）。

---

## 📦 安装指南

### 🚨 环境要求

- Python `3.8` 或以上版本
- 推荐使用虚拟环境（`venv`、`conda`）

### 🧱 安装步骤

1. 安装 PyTorch（建议使用CUDA支持版，如果没有GPU，可跳过此步骤）:

   ```bash
   # 请根据你的系统和CUDA版本，从 https://pytorch.org 获取合适的安装命令
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

2. 安装其他依赖包：

   ```bash
   pip install demucs PyQt5 librosa numpy mido scipy matplotlib Pillow
   ```

---

## 🚀 如何运行

确保你已安装好依赖，然后在终端中运行：

```bash
python your_script_filename.py
```

默认会弹出一个图形界面窗口，包含两个主要功能：

- **音频转MIDI**
- **MIDI转WAV**

---

## 📂 项目结构（单文件版）

```text
piano_singing_toolkit_v7.7.py     # 主程序（图形界面 + GUI逻辑 + 数据处理）
README.md                         # 项目说明
requirements.txt                  # （可选）依赖项列表
```

---

## 🛠️ 功能对比表

| 功能                     | 音轨分离模式 | 整体EQ模式 |
|--------------------------|----------------|----------------|
| GPU加速支持              | ✅              | ❌              |
| Demucs人声/鼓分离         | ✅              | ❌              |
| 手动EQ调节              | ❌              | ✅              |
| 各音轨个性敏感度调整     | ✅              | ❌              |
| 生成可视化频谱图         | ✅（可选）      | ❌              |

---

## 📑 使用提示

- **频谱图透明合成修复**：采用Matplotlib设置Figure和Axes透明，实现alpha叠加。
- **音轨增益敏感度调节**：通过滑块+数值框调整音量（影响音符侦测数量），更精确控制MIDI输出。
- **缓存保护**：音轨分离结果保存在 `./separated_demucs/` 下，避免频繁重复运行 Demucs。
- **MIDI合成器**：基于Sine wave 定制合成，支持 Polyphony 多音轨叠加。

---

## 🎯 示例用途

- 将人声/伴奏转为MIDI进行音符分析
- 从鼓点、贝斯中提取节奏信息
- 对MIDI进行简易演奏合成演示
- 频谱图可用于研究乐器分布或作为视觉素材

---

## 📜 许可协议

本项目使用 **MIT License** 开源。

---

## 🤝 交流与贡献

欢迎提交 Issues 或 Pull Request：
- 发现Bug
- 提出功能建议
- 优化算法
- 翻译界面 / 添加多语言（未来支持）

---

## 💡 未来计划（TODO）

- ✅ 多线程渲染
- ✅ 多音轨支持
- 🚧 替换合成器为真实音源（SoundFont）
- 🚧 LLM自动识别乐器段落
- 🚧 添加控制台命令行模式

---

项目由开源爱好者开发，如对你有帮助，欢迎点个⭐支持！

```

---

### ✅ 如果你有 `requirements.txt`，别忘了附带：

```
demucs
torch
PyQt5
librosa
numpy
mido
scipy
matplotlib
Pillow
```
