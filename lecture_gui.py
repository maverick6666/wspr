# -*- coding: utf-8 -*-
"""
강의 녹음 -> 텍스트 변환 GUI

- 시스템 오디오(스피커 출력)를 loopback 으로 캡처
- 60초 청크로 저장하면서 faster-whisper(small, int8) 로 한국어 변환
- 결과는 {강의이름}.txt 에 누적, 오디오는 audio/{강의이름}/ 에 저장

transcribe_lecture.py(CLI) 의 GUI 버전. '강의녹음.bat' 더블클릭으로 실행.
"""
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc
from scipy.io import wavfile

# ===== 설정 (CLI 버전과 동일) =====
SAMPLE_RATE = 16000
CHUNK_SECONDS = 60
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
LANGUAGE = "ko"
BEAM_SIZE = 5
READ_SECONDS = 0.2          # 레벨 미터 갱신 주기
LEVEL_DECAY = 0.85          # 레벨 미터 감쇠 (1에 가까울수록 천천히 내려옴)

BASE_DIR = Path(__file__).parent
LEVEL_W = 480
LEVEL_H = 26
FONT = ("Malgun Gothic", 10)
FONT_B = ("Malgun Gothic", 11, "bold")


class App:
    def __init__(self, root):
        self.root = root

        # 스레드 <-> GUI 통신용 큐
        self.audio_queue = queue.Queue()   # 변환 대기 wav 경로
        self.log_queue = queue.Queue()     # 로그 메시지
        self.level_queue = queue.Queue()   # 입력 레벨(peak)

        # 상태
        self.app_running = True
        self.recording = False
        self.model = None
        self.model_ready = False
        self.chunk_count = 0
        self.audio_dir = None
        self.transcript_file = None
        self._level_disp = 0.0   # 레벨 미터 표시값 (감쇠 적용된 peak)

        self._build_ui()

        # 백그라운드 스레드: 오디오 모니터, 변환 워커, 모델 로더
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()
        threading.Thread(target=self._load_model, daemon=True).start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll)

    # ---------------- UI ----------------
    def _build_ui(self):
        self.root.title("강의 녹음 → 텍스트")
        self.root.geometry("540x650")
        self.root.configure(bg="#f4f4f4")

        tk.Label(self.root, text="강의 녹음 → 텍스트 변환",
                 font=("Malgun Gothic", 14, "bold"), bg="#f4f4f4").pack(anchor="w", padx=14, pady=(12, 4))

        # 강의 이름
        frm = tk.Frame(self.root, bg="#f4f4f4")
        frm.pack(fill="x", padx=14, pady=6)
        tk.Label(frm, text="강의 이름:", font=FONT, bg="#f4f4f4").pack(side="left")
        self.name_var = tk.StringVar(value="정보보안 9주차 -3")
        self.name_entry = tk.Entry(frm, textvariable=self.name_var, font=FONT)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # 입력 레벨 미터
        tk.Label(self.root, text="입력 레벨  —  녹음 전, 영상을 틀어 초록색이 되는지 확인하세요",
                 font=FONT, bg="#f4f4f4").pack(anchor="w", padx=14, pady=(12, 0))
        self.level_canvas = tk.Canvas(self.root, width=LEVEL_W, height=LEVEL_H,
                                      bg="#e2e2e2", highlightthickness=1, highlightbackground="#aaa")
        self.level_canvas.pack(padx=14, pady=(2, 0))
        self.level_var = tk.StringVar(value="입력 레벨:   0.0%")
        tk.Label(self.root, textvariable=self.level_var, font=FONT, bg="#f4f4f4").pack(anchor="w", padx=14)

        # 시작 / 중지 버튼
        frm_btn = tk.Frame(self.root, bg="#f4f4f4")
        frm_btn.pack(fill="x", padx=14, pady=10)
        self.start_btn = tk.Button(frm_btn, text="●  녹음 시작", font=FONT_B, bg="#33aa55", fg="white",
                                   width=15, height=2, relief="flat", command=self._start)
        self.start_btn.pack(side="left", padx=(0, 10))
        self.stop_btn = tk.Button(frm_btn, text="■  중지", font=FONT_B, bg="#cc4444", fg="white",
                                  width=15, height=2, relief="flat", state="disabled", command=self._stop)
        self.stop_btn.pack(side="left")

        # 상태
        self.status_var = tk.StringVar(value="상태: 모델 로딩 중...")
        tk.Label(self.root, textvariable=self.status_var, font=FONT_B, bg="#f4f4f4", fg="#333").pack(
            anchor="w", padx=14, pady=(0, 6))

        # 진행 로그
        tk.Label(self.root, text="진행 상황:", font=FONT, bg="#f4f4f4").pack(anchor="w", padx=14)
        self.log = scrolledtext.ScrolledText(self.root, height=12, font=("Consolas", 9),
                                             state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=14, pady=(2, 6))

        # 결과 열기
        frm_open = tk.Frame(self.root, bg="#f4f4f4")
        frm_open.pack(fill="x", padx=14, pady=(0, 12))
        tk.Button(frm_open, text="결과 폴더 열기", font=FONT, command=self._open_folder).pack(side="left")
        tk.Button(frm_open, text="텍스트 파일 열기", font=FONT, command=self._open_text).pack(side="left", padx=8)

    # ---------------- 오디오 모니터 + 녹음 ----------------
    def _monitor_loop(self):
        try:
            speaker = sc.default_speaker()
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
            self.log_queue.put(f"[오디오] 디바이스: {speaker.name}")
        except Exception as e:
            self.log_queue.put(f"[오류] 오디오 장치 초기화 실패: {e}")
            return

        frames = int(SAMPLE_RATE * READ_SECONDS)
        target = int(SAMPLE_RATE * CHUNK_SECONDS)
        buf, acc = [], 0
        try:
            with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                while self.app_running:
                    data = rec.record(numframes=frames).flatten()
                    peak = float(np.max(np.abs(data))) if data.size else 0.0
                    self.level_queue.put(peak)

                    if self.recording:
                        buf.append(data)
                        acc += data.size
                        if acc >= target:
                            self._save_chunk(np.concatenate(buf))
                            buf, acc = [], 0
                    elif buf:
                        # 방금 중지됨 -> 3초 이상 남았으면 마지막 청크로 저장
                        if acc >= SAMPLE_RATE * 3:
                            self._save_chunk(np.concatenate(buf))
                        buf, acc = [], 0
        except Exception as e:
            self.log_queue.put(f"[오류] 녹음 루프 중단: {e}")

    def _save_chunk(self, samples):
        try:
            audio_i16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = self.audio_dir / f"chunk_{ts}.wav"
            wavfile.write(str(wav_path), SAMPLE_RATE, audio_i16)
            self.audio_queue.put(wav_path)
            self.chunk_count += 1
            self.log_queue.put(f"[녹음] 청크 #{self.chunk_count} 저장 ({ts})")
        except Exception as e:
            self.log_queue.put(f"[오류] 청크 저장 실패: {e}")

    # ---------------- 변환 ----------------
    def _load_model(self):
        self.log_queue.put(f"[모델] {MODEL_SIZE}({COMPUTE_TYPE}) 로딩 중... 잠시만 기다려 주세요")
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
            self.model_ready = True
            self.log_queue.put("[모델] 로딩 완료 ✓  이제 녹음할 수 있어요")
        except Exception as e:
            self.log_queue.put(f"[오류] 모델 로딩 실패: {e}")

    def _transcribe_loop(self):
        while self.app_running and not self.model_ready:
            time.sleep(0.3)
        while self.app_running:
            try:
                wav_path = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.model is None:
                continue
            try:
                segments, _ = self.model.transcribe(
                    str(wav_path), language=LANGUAGE, beam_size=BEAM_SIZE,
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
                text = "".join(s.text for s in segments).strip()
            except Exception as e:
                self.log_queue.put(f"[오류] 변환 실패 {wav_path.name}: {e}")
                continue

            if text:
                ts = wav_path.stem.replace("chunk_", "")
                try:
                    with open(self.transcript_file, "a", encoding="utf-8") as f:
                        f.write(f"\n[{ts}]\n{text}\n")
                except Exception as e:
                    self.log_queue.put(f"[오류] 파일 쓰기 실패: {e}")
                    continue
                preview = text[:38].replace("\n", " ")
                self.log_queue.put(f"[변환] {preview}...")
            else:
                self.log_queue.put("[변환] (무음 청크 건너뜀)")

    # ---------------- 버튼 핸들러 ----------------
    def _start(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("이름 필요", "강의 이름을 입력해 주세요.")
            return
        if set('<>:"/\\|?*') & set(name):
            messagebox.showwarning("이름 오류", '파일명에 쓸 수 없는 문자가 있어요:  < > : " / \\ | ? *')
            return
        if not self.model_ready:
            if not messagebox.askyesno(
                    "모델 로딩 중",
                    "변환 모델이 아직 로딩 중입니다.\n녹음은 시작되지만 변환은 모델 준비 후 진행됩니다.\n계속할까요?"):
                return

        self.transcript_file = BASE_DIR / f"{name}.txt"
        self.audio_dir = BASE_DIR / "audio" / name
        try:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("오류", f"폴더 생성 실패: {e}")
            return

        self.chunk_count = 0
        self.recording = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.name_entry.config(state="disabled")
        self.status_var.set(f"상태: ● 녹음 중 — {name}")
        self.log_queue.put(f"===== 녹음 시작: {name} =====")
        self.log_queue.put("팁: 영상이 소리나는 상태에서 시작하세요. 위 레벨이 초록이면 잘 잡힙니다.")

    def _stop(self):
        self.recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.name_entry.config(state="normal")
        self.status_var.set("상태: ■ 중지됨 (남은 청크 변환 중)")
        self.log_queue.put("===== 녹음 중지 — 남은 청크 변환 처리 중 =====")

    def _open_folder(self):
        try:
            os.startfile(str(BASE_DIR))
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _open_text(self):
        if self.transcript_file and Path(self.transcript_file).exists():
            try:
                os.startfile(str(self.transcript_file))
            except Exception as e:
                messagebox.showerror("오류", str(e))
        else:
            messagebox.showinfo("안내", "아직 결과 텍스트 파일이 없습니다.\n녹음을 먼저 진행하세요.")

    # ---------------- 폴링 (메인 스레드에서만 위젯 갱신) ----------------
    def _poll(self):
        # 이번 주기에 쌓인 peak 중 최대값을 취하고, 감쇠를 적용해 부드럽게 표시
        peaks = []
        try:
            while True:
                peaks.append(self.level_queue.get_nowait())
        except queue.Empty:
            pass
        raw = max(peaks) if peaks else 0.0
        self._level_disp = max(raw, self._level_disp * LEVEL_DECAY)
        self._draw_level(self._level_disp)

        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass

        if self.model_ready and not self.recording and self.status_var.get().endswith("로딩 중..."):
            self.status_var.set("상태: 대기 중 — '녹음 시작'을 누르세요")

        if self.app_running:
            self.root.after(100, self._poll)

    def _draw_level(self, level):
        # soundcard 는 -1.0~1.0 float 를 반환 → 그대로 0~100% 로 환산
        pct = max(0.0, min(100.0, level * 100.0))
        fill = int(LEVEL_W * pct / 100.0)
        if pct < 2:
            color, label = "#cc3333", "소리 없음 / 너무 작음"
        elif pct < 15:
            color, label = "#d4a017", "낮음 — 볼륨 올리세요"
        else:
            color, label = "#2e9e3f", "양호"
        self.level_canvas.delete("bar")
        if fill > 0:
            self.level_canvas.create_rectangle(0, 0, fill, LEVEL_H, fill=color, width=0, tags="bar")
        self.level_var.set(f"입력 레벨: {pct:5.1f}%    ({label})")

    def _append_log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ---------------- 종료 ----------------
    def _on_close(self):
        if self.recording:
            if not messagebox.askyesno("종료 확인", "녹음 중입니다. 정말 종료할까요?"):
                return
        self.app_running = False
        self.recording = False
        self.root.after(250, self.root.destroy)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
