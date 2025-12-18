# Auto-translate with Whisper and NLLB

一个 **本地部署的语音翻译项目**，基于 **Whisper（ASR） + NLLB（机器翻译）**，支持实时麦克风输入和音频文件翻译，支持 CPU / GPU。

---

## 功能简介

- 🎙️ 语音转文字（Whisper / faster-whisper）
- 🌍 多语言翻译（Facebook NLLB-200）
- 🧠 全程本地运行，无需在线 API
- ⚡ 支持 GPU 加速（可选）

---

## 项目结构

```text
Auto-translate-with-Whisper-and-NLLB/
├── src/
│   ├── autotranslate.py      # 实时音频翻译
│   ├── mp3translate.py       # 音频文件翻译
│   └── __init__.py
├── models/
│   └── README.md             # 模型下载说明（模型不进仓库）
├── requirements.txt          # Python 依赖（精简版）
├── .gitignore
└── README.md
环境要求
Python >= 3.9

推荐使用 Conda 或 venv

（可选）NVIDIA GPU + CUDA

安装依赖（Libraries）
1️⃣ 创建虚拟环境（推荐）
Conda
bash
复制代码
conda create -n translate python=3.10
conda activate translate
venv
bash
复制代码
python -m venv .venv
source .venv/bin/activate
2️⃣ 安装 Python 依赖
bash
复制代码
pip install -r requirements.txt
requirements.txt 示例
text
复制代码
torch
faster-whisper
transformers
ctranslate2
numpy
sounddevice
模型管理（Models）【重要】
⚠️ 模型体积较大，不直接提交到 GitHub

本项目采用 代码与模型分离 的方式管理模型。

Whisper 模型（ASR）
使用 faster-whisper

首次运行会自动下载

默认下载目录：models/

示例代码：

python
复制代码
WhisperModel("small", download_root="models")
可选模型：

tiny

base

small（推荐）

medium

large-v3

NLLB 翻译模型（MT）
推荐模型：

facebook/nllb-200-distilled-600M

手动下载（推荐）
使用 HuggingFace CLI：

bash
复制代码
pip install huggingface-hub
hf download facebook/nllb-200-distilled-600M \
  --local-dir models/nllb \
  --local-dir-use-symlinks False
或使用 Python 脚本：

python
复制代码
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/nllb-200-distilled-600M",
    local_dir="models/nllb",
    local_dir_use_symlinks=False
)
模型目录结构示例
text
复制代码
models/
├── whisper-small/
└── nllb/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json
运行项目
实时麦克风翻译
bash
复制代码
python src/autotranslate.py
翻译音频文件
bash
复制代码
python src/mp3translate.py input.mp3
CPU / GPU 说明
CPU：无需额外配置，速度较慢

GPU：

安装 CUDA 对应版本的 PyTorch

faster-whisper 会自动使用 GPU

检查 GPU 是否可用：

python
复制代码
import torch
print(torch.cuda.is_available())
.gitignore 示例
gitignore
复制代码
models/
*.bin
*.pt
__pycache__/
.venv/
常见问题
Q: 为什么不把模型直接提交到 GitHub？
A: 模型体积大，GitHub 有大小限制，不利于维护。

Q: 第一次运行很慢？
A: 正在下载模型，属于正常现象。

License
MIT License

作者
Kun / Zhangkunzhan
With the help of chatgpt and doubao
