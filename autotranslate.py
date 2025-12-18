import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import ctranslate2

# ================== 音频参数 ==================
SAMPLE_RATE = 16000
RECORD_DURATION = 1.0
CHANNELS = 1

# ================== 模型路径 ==================
WHISPER_MODEL_PATH = "/home/kun/models/faster-whisper-small"
MARIAN_MODEL_PATH = "/home/kun/models/marian-en-zh-ct2"

print("运行设备：CPU")

# ================== Whisper ASR ==================
asr_model = WhisperModel(
    WHISPER_MODEL_PATH,
    device="cpu",
    compute_type="int8",
    cpu_threads=8
)

# ================== Marian 翻译 ==================
translator = ctranslate2.Translator(
    MARIAN_MODEL_PATH,
    device="cpu",
    compute_type="int8",
    inter_threads=2,
    intra_threads=8
)

# ================== 音频采集 ==================
def record_audio():
    audio = sd.rec(
        int(RECORD_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32
    )
    sd.wait()
    return audio.squeeze()

# ================== ASR ==================
def asr(audio):
    segments, _ = asr_model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        without_timestamps=True
    )
    return " ".join(s.text.strip() for s in segments)

# ================== 翻译 ==================
def translate(text):
    if not text:
        return ""
    tokens = text.strip().split()
    result = translator.translate_batch(
        [tokens],
        beam_size=4
    )
    return " ".join(result[0].hypotheses[0])

print("开始 CPU 实时翻译，Ctrl+C 退出")

try:
    while True:
        audio = record_audio()
        en = asr(audio)
        if not en:
            continue
        zh = translate(en)
        print("EN:", en)
        print("ZH:", zh)
        print("-" * 40)
except KeyboardInterrupt:
    print("退出")
