"""
강의 실시간 녹음 + Whisper 변환 스크립트.

- Windows 기본 스피커의 loopback으로 시스템 오디오 캡처
- CHUNK_SECONDS 단위로 잘라 wav 저장
- faster-whisper base(int8)로 텍스트 변환해 {name}.txt 에 누적
- STOP 파일 생성 or Ctrl+C로 종료 (큐에 남은 청크까지 모두 변환 후 종료)

사용:
    py transcribe_lecture.py --name "7주차_CH5_CFG_1"
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel
from scipy.io import wavfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# ===== 설정 =====
SAMPLE_RATE = 16000
CHUNK_SECONDS = 60
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
LANGUAGE = "ko"
BEAM_SIZE = 5

BASE_DIR = Path(__file__).parent

audio_queue: "queue.Queue[Path]" = queue.Queue()
stop_event = threading.Event()


def record_loop(audio_dir: Path) -> None:
    speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    print(f"[녹음] 디바이스: {speaker.name}")

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while not stop_event.is_set():
            chunk_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
            data = recorder.record(numframes=chunk_samples)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = audio_dir / f"chunk_{timestamp}.wav"

            audio_int16 = np.clip(data.flatten() * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(str(wav_path), SAMPLE_RATE, audio_int16)

            audio_queue.put(wav_path)
            print(f"[녹음] 저장: {wav_path.name} (큐: {audio_queue.qsize()})")


def transcribe_loop(transcript_file: Path) -> None:
    print(f"[변환] 모델 로딩: {MODEL_SIZE} ({COMPUTE_TYPE})")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    print("[변환] 모델 로딩 완료. 대기 중...\n")

    while not (stop_event.is_set() and audio_queue.empty()):
        try:
            wav_path = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        print(f"[변환] 시작: {wav_path.name}")
        segments, _info = model.transcribe(
            str(wav_path),
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = "".join(seg.text for seg in segments).strip()

        if text:
            timestamp = wav_path.stem.replace("chunk_", "")
            with open(transcript_file, "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}]\n{text}\n")
            preview = text[:60].replace("\n", " ")
            print(f"[변환] 완료 ({len(text)}자): {preview}...\n")
        else:
            print(f"[변환] 무음/인식 실패: {wav_path.name}\n")

        audio_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="강의 실시간 녹음 + Whisper 변환")
    parser.add_argument("--name", required=True, help='강의 이름 (예: "7주차_CH5_CFG_1")')
    parser.add_argument(
        "--out-dir",
        default=None,
        help="결과물 저장 폴더 (기본: 스크립트가 있는 폴더)",
    )
    args = parser.parse_args()

    name = args.name.strip()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else BASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = out_dir / f"{name}.txt"
    audio_dir = out_dir / "audio" / name
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f" 강의: {name}")
    print(f" 스크립트 파일: {transcript_file.name}")
    print(f" 청크 길이: {CHUNK_SECONDS}초")
    print(f" 모델: {MODEL_SIZE} ({COMPUTE_TYPE})")
    print(" 종료: STOP 파일 생성 또는 Ctrl+C")
    print("=" * 50 + "\n")

    rec_thread = threading.Thread(target=record_loop, args=(audio_dir,), daemon=True)
    trans_thread = threading.Thread(target=transcribe_loop, args=(transcript_file,), daemon=True)
    rec_thread.start()
    trans_thread.start()

    stop_flag = BASE_DIR / "STOP"
    if stop_flag.exists():
        stop_flag.unlink()

    try:
        while not stop_flag.exists():
            time.sleep(1)
        print("\n[종료] STOP 파일 감지. 남은 큐 처리 중 (변환 끝날 때까지 대기)...")
    except KeyboardInterrupt:
        print("\n[종료] 중단 신호 수신. 남은 큐 처리 중 (변환 끝날 때까지 대기)...")

    stop_event.set()
    trans_thread.join(timeout=600)
    try:
        stop_flag.unlink()
    except FileNotFoundError:
        pass
    print(f"\n[종료] 완료. 스크립트 파일: {transcript_file}")


if __name__ == "__main__":
    main()
