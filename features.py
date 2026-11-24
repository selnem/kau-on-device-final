"""공용 전처리: 16 kHz mono 1초 오디오 -> uint8 [3, 64, 64]

학습(Colab), 보정 이미지 생성, 보드(Pi 5) 추론이 모두 이 파일 하나를 쓴다.
전처리가 조금만 달라도 보드 정확도가 떨어지므로, 보드에는 이 파일을 그대로 복사한다.
numpy만 사용한다 (Pi에 librosa를 깔지 않기 위해).

출력 배열: [채널, Mel, 시간]
  채널 0 = log-Mel (소리 세기), 1 = 시간 방향 1차 변화량, 2 = 2차 변화량
  Mel   행 0 = 고주파, 행 63 = 저주파 (이미지로 볼 때 아래가 저음)
  시간  열 0 = 1초 전, 열 63 = 지금

아래 상수를 바꾸면 모델을 다시 학습하고 보드 쪽 파일도 바꿔야 한다.

사용 예:
    from features import features
    img = features(audio_1s)          # audio_1s: 길이 16000, float(-1~1) 또는 int16
    img.shape, img.dtype              # (3, 64, 64) uint8
"""
import numpy as np

# ---------------------------------------------------------------- 규격
SR = 16000
N_SAMPLES = SR                 # 1초
WIN, N_FFT, HOP = 400, 512, 250  # 25 ms 창, 15.6 ms 간격 -> 1초를 64칸으로
N_FRAMES = 64
N_MELS, FMIN, FMAX = 64, 50.0, 8000.0  # 휘파람 1~3 kHz, '쉬' 2~8 kHz가 모두 들어감
DELTA_N = 4                    # 변화량은 앞뒤 4프레임(폭 9)으로 계산

# 0~255로 바꿀 때 쓰는 고정 범위. 샘플마다 최소·최대로 맞추면 조용한 배경의 잡음이
# 큰 소리처럼 늘어나므로, 학습 데이터에서 한 번 재서 고정한다 (feature_ranges.py).
# 11/19 공개 데이터(background·speech·shh 1초 구간 536개) 측정값:
#   dB p1 -89.5 / p99 1.9,  |1차| p99 6.06,  |2차| p99 1.41
# 위쪽은 마이크 가까이에서 낸 큰 소리가 포화되지 않게 약간 여유를 둠.
# 자체 녹음(박수·혀 차기처럼 급한 소리)이 들어오면 다시 재서 확인할 것.
DB_LO, DB_HI = -90.0, 5.0      # log-Mel dB
D1_MAX = 6.0                   # 1차 변화량 |값| 상한 (dB/프레임)
D2_MAX = 1.5                   # 2차 변화량 |값| 상한


# ---------------------------------------------------------------- Mel 필터 (librosa 기본값과 같은 Slaney 방식)
def _hz_to_mel(f):
    # 1 kHz 아래는 직선, 위는 로그 눈금
    f = np.asarray(f, np.float64)
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_hz / f_sp + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
                    f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, np.float64)
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def _mel_filters():
    """[64 Mel, 257 FFT칸] 삼각형 필터. 대역마다 면적이 같도록 정규화."""
    fft_hz = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    width = np.diff(edges)
    ramps = edges[:, None] - fft_hz[None, :]
    rise = -ramps[:-2] / width[:-1, None]
    fall = ramps[2:] / width[1:, None]
    w = np.maximum(0, np.minimum(rise, fall))
    w *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    return w.astype(np.float32)


def _window():
    """길이 400 Hann 창을 512 가운데에 둔다 (librosa와 같은 배치)."""
    n = np.arange(WIN)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / WIN)
    pad = (N_FFT - WIN) // 2
    return np.pad(w, (pad, N_FFT - WIN - pad)).astype(np.float32)


# 한 번만 만들어 두고 계속 쓴다
_MEL = _mel_filters()
_WIN = _window()
_IDX = np.arange(N_FRAMES)[:, None] * HOP + np.arange(N_FFT)[None, :]  # 프레임별 샘플 위치


# ---------------------------------------------------------------- 단계별 함수
def to_float(x):
    """int16 마이크 입력도 받을 수 있게 -1~1 float32로 바꾼다."""
    x = np.asarray(x)
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def fit_length(x, n=N_SAMPLES):
    """길이를 1초로 맞춘다. 짧으면 앞뒤를 0으로 채워 가운데 두고, 길면 가운데를 자른다."""
    x = to_float(x)
    if len(x) < n:
        left = (n - len(x)) // 2
        return np.pad(x, (left, n - len(x) - left))
    start = (len(x) - n) // 2
    return x[start:start + n]


def mel_power(x):
    """1초 오디오 -> Mel 파워 [64 Mel(행 0 = 저주파), 64 시간]."""
    x = to_float(x)
    assert x.shape == (N_SAMPLES,), f"16000 샘플(1초)이 필요함, 입력 {x.shape}"
    frames = np.pad(x, N_FFT // 2)[_IDX] * _WIN     # 앞뒤 256샘플 0 패딩 후 64프레임 자르기
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # [64 시간, 257 주파수]
    return (power @ _MEL.T).T                         # [64 Mel, 64 시간]


def log_mel(x):
    """dB 단위 log-Mel. 1e-10은 log(0)을 막기 위한 값."""
    return 10.0 * np.log10(mel_power(x) + 1e-10)


def delta(c):
    """시간 방향 변화량. 앞뒤 4프레임의 기울기(최소제곱 직선의 기울기)를 쓴다.
    끝부분은 마지막 값을 반복해 채운다 (librosa delta(width=9, mode='nearest')와 같음)."""
    k = np.arange(1, DELTA_N + 1)
    p = np.pad(c, ((0, 0), (DELTA_N, DELTA_N)), mode="edge")
    t = c.shape[1]
    d = sum(i * (p[:, DELTA_N + i:DELTA_N + i + t] - p[:, DELTA_N - i:DELTA_N - i + t]) for i in k)
    return d / (2 * np.sum(k ** 2))


def _to_u8(v, lo, hi):
    return np.round(np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def features(x):
    """1초 오디오 -> uint8 [3, 64, 64]. 학습·보정·보드 추론 모두 이 함수 하나를 쓴다."""
    db = log_mel(x)
    d1 = delta(db)   # 휘파람 ↗/↘처럼 음 높이가 움직이는 방향
    d2 = delta(d1)   # 혀 차기·박수처럼 짧고 급한 변화
    img = np.stack([_to_u8(db, DB_LO, DB_HI),
                    _to_u8(d1, -D1_MAX, D1_MAX),
                    _to_u8(d2, -D2_MAX, D2_MAX)])
    return np.ascontiguousarray(img[:, ::-1, :])  # 저주파를 아래로 (이미지로 볼 때 보기 편하게)
