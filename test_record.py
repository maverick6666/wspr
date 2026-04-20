"""10초 녹음 후 Whisper로 변환하는 테스트 스크립트."""

from pathlib import Path

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel
from scipy.io import wavfile

DURATION = 10
SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).parent
OUT_WAV = BASE_DIR / "test_sample.wav"

speaker = sc.default_speaker()
print(f"[디바이스] {speaker.name}")

mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
print(f"[녹음] {DURATION}초 녹음 시작...")

with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
    data = recorder.record(numframes=SAMPLE_RATE * DURATION)

audio = data.flatten()
rms = float(np.sqrt(np.mean(audio ** 2)))
peak = float(np.max(np.abs(audio)))
print(f"[볼륨] RMS={rms:.4f}, Peak={peak:.4f}")
print("       (RMS < 0.001 이면 소리 거의 안 잡힘)")

audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
wavfile.write(str(OUT_WAV), SAMPLE_RATE, audio_int16)
print(f"[저장] {OUT_WAV.name}")

print("[변환] 모델 로딩 중...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("[변환] Whisper 돌리는 중...")
segments, info = model.transcribe(
    str(OUT_WAV), language="ko", beam_size=1, vad_filter=True
)
text = "".join(s.text for s in segments).strip()

print("\n" + "=" * 50)
print(f"감지 언어: {info.language} (확률 {info.language_probability:.2f})")
print(f"변환 결과:\n{text if text else '(무음/인식 실패)'}")
print("=" * 50)
