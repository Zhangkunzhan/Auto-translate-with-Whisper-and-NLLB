#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CET6 离线语音识别 + 逐段翻译（Whisper + NLLB 终版）
完全离线 | 逐段 EN → ZH | 稳定不截断
"""

import os
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================== 路径配置 ==================
WHISPER_MODEL_DIR = "/home/kun/models/faster-whisper-small"
NLLB_MODEL_DIR = "/home/kun/models/nllb-200"
AUDIO_PATH = "cet6_2025_06_1.mp3"

# NLLB 语言代码（非常关键）
SRC_LANG = "eng_Latn"
TGT_LANG = "zho_Hans"

# ================== Whisper ==================
def load_whisper():
    print("加载 Whisper 模型...")
    model = WhisperModel(
        WHISPER_MODEL_DIR,
        device="cpu",
        compute_type="int8"
    )
    return model


def transcribe_audio(model, audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    print(f"识别语言: {info.language}")
    print("开始语音识别...\n")

    results = []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        if text:
            results.append(text)
            print(f"[EN {i:03d}] {text}")

    return results


# ================== NLLB ==================
def load_nllb():
    print("\n加载 NLLB 翻译模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        NLLB_MODEL_DIR,
        local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_DIR,
        local_files_only=True,
        dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


def translate_segment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    forced_bos_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            max_new_tokens=256
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ================== 主流程 ==================
def main():
    print("=" * 60)
    print("CET6 离线语音识别 + 逐段翻译工具（终版）")
    print("=" * 60)

    if not os.path.exists(AUDIO_PATH):
        print(f"音频文件不存在: {AUDIO_PATH}")
        return

    whisper = load_whisper()
    segments = transcribe_audio(whisper, AUDIO_PATH)

    tokenizer, nllb = load_nllb()

    print("\n" + "-" * 50)
    print("开始逐段翻译...\n")

    for i, en in enumerate(segments, 1):
        zh = translate_segment(en, tokenizer, nllb)
        print(f"[EN {i:03d}] {en}")
        print(f"[ZH {i:03d}] {zh}\n")

    print("=" * 60)
    print("翻译完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
