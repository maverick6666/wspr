# wspr — 강의 실시간 녹음 + Whisper 텍스트 변환

인강/유튜브 강의 같은 **스트리밍 오디오를 실시간으로 녹음**하고, 백그라운드에서 **faster-whisper로 텍스트 변환**해 강의별 `.txt` 파일로 누적 저장하는 스크립트입니다.

변환된 원문은 완벽하진 않지만, **Claude Code 같은 AI에 맡겨 교정·요약**하면 강의노트 보완용으로 훌륭하게 쓸 수 있습니다.

---

## 동작 방식

```
[시스템 오디오 loopback]
      ↓
[60초 청크로 wav 저장]
      ↓
[faster-whisper (small, int8, beam_size=5)]
      ↓
[{강의이름}.txt 에 타임스탬프와 함께 누적]
```

- 스피커든 블루투스 이어폰이든 **윈도우 기본 출력 장치**로 나오는 소리를 잡습니다 (가상 케이블 불필요).
- VAD(Voice Activity Detection) 내장: 무음 구간은 자동 스킵.
- 종료 시 큐에 남은 청크까지 전부 변환한 뒤 안전하게 마칩니다.

---

## 요구사항

- Windows 10/11
- Python 3.10+
- ffmpeg
- 공간: small 모델 ~450MB 캐시

---

## 설치

### 1. Python / ffmpeg

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

설치 후 **새 PowerShell 창**을 여세요 (PATH 반영).

### 2. Python 패키지

```powershell
cd <이 저장소 경로>
py -m pip install -r requirements.txt
```

첫 실행 시 `small` 모델(~450MB)이 자동 다운로드됩니다. 미리 받고 싶으면:

```powershell
py -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

---

## 사용법

### 방법 A — 직접 실행 (수동)

```powershell
py transcribe_lecture.py --name "7주차_CH5_CFG_1"
```

- 강의 재생 시작 → 60초마다 청크 저장 + 변환 로그 출력
- 종료: `Ctrl+C` 또는 같은 폴더에 빈 `STOP` 파일 생성
- 결과물: `7주차_CH5_CFG_1.txt`

### 방법 B — Claude Code에서 "시작/끝"으로 제어 (추천)

Claude Code(VS Code 익스텐션 등)를 열어놓고 아래처럼 대화하면 됩니다.

**시작하기**

> `CFG_1 시작` 또는 `7주차 CFG 1 강의 녹음 시작해줘`

Claude가 백그라운드로 `transcribe_lecture.py --name "..."` 을 실행합니다.

**끝내기**

> `끝` 또는 `녹음 종료해줘`

Claude가 `STOP` 파일을 생성 → 큐에 남은 청크까지 변환 끝난 뒤 안전하게 종료합니다.

**변환 후 교정/요약**

> `7주차_CH5_CFG_1.txt 읽고 오타·문맥 교정해서 clean_7주차_CH5_CFG_1.txt로 저장해줘`
> `이번 강의 핵심 5줄로 요약해줘`

원문은 거칠어도 Claude가 문맥 보고 고쳐주면 꽤 쓸만해집니다.

---

## 설정 변경

`transcribe_lecture.py` 상단 상수를 바꾸면 됩니다.

| 상수 | 기본값 | 설명 |
|---|---|---|
| `CHUNK_SECONDS` | 60 | 청크 저장 주기 (짧을수록 변환 지연 ↓, 문맥 ↓) |
| `MODEL_SIZE` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `COMPUTE_TYPE` | `int8` | `int8` / `int8_float16` / `float16` / `float32` |
| `BEAM_SIZE` | 5 | 크면 정확도 ↑ 속도 ↓ (1~5 권장) |

### 모델별 대략 기준 (CPU 기준)

| 모델 | RAM | 1분 오디오 변환 | 품질 |
|---|---|---|---|
| tiny | ~150MB | ~10초 | 낮음 |
| base | ~300MB | ~20초 | 보통 |
| **small** | **~500MB** | **~45초** | **좋음 (추천)** |
| medium | ~1.5GB | ~2분 | 더 좋음 |

---

## 배속 재생 팁

강의를 **1.5배속**으로 듣는 걸 추천합니다.
- 2배속 이상은 한국어 인식률이 눈에 띄게 떨어짐
- 1.5배는 체감상 거의 동일한 품질
- 전체 작업 시간 자동 단축

---

## 결과물 구조

```
wspr/
├── transcribe_lecture.py
├── test_record.py             # 10초 짧은 테스트 스크립트
├── requirements.txt
├── {강의이름}.txt              # ← 누적 변환 결과 (gitignore)
└── audio/
    └── {강의이름}/
        └── chunk_YYYYMMDD_HHMMSS.wav  # ← 원본 녹음 (gitignore)
```

`.txt` 포맷 예시:

```
[20260420_192031]
안녕하세요. 오늘은 문맥 자유 문법에 대해 살펴보겠습니다...

[20260420_192131]
CFG는 다음과 같이 정의되는데요...
```

---

## 테스트 스크립트

오디오 캡처가 제대로 되는지 10초 만에 확인할 수 있습니다.

```powershell
py test_record.py
```

출력 예:

```
[디바이스] 스피커(AirPods Pro)
[녹음] 10초 녹음 시작...
[볼륨] RMS=0.0814, Peak=0.5164
       (RMS < 0.001 이면 소리 거의 안 잡힘)
[변환] Whisper 돌리는 중...
감지 언어: ko (확률 1.00)
변환 결과:
...
```

RMS가 0에 가까우면 윈도우 기본 출력 장치를 확인하세요.

---

## 문제 해결

**"ffmpeg를 찾을 수 없음" 에러**
→ ffmpeg 설치 후 PowerShell 새로 열었는지 확인

**변환 텍스트가 계속 비어 있음**
→ `test_record.py` 돌려서 RMS 체크. 0에 가까우면 기본 출력 장치 잘못 잡힘

**RAM 부족으로 느려짐**
→ `MODEL_SIZE = "base"` 로 다운그레이드

**터미널에 한글이 깨져 보임**
→ 실제 `.txt` 파일은 UTF-8로 정상 저장됨. 메모장·VS Code로 열면 OK

---

## 라이선스

자유롭게 사용.
