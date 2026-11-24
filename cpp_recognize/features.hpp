#pragma once

#include <cstdint>

// features.py 와 같은 값.
// audio: 16 kHz, 1초, float [-1, 1], 길이 16000
// out_hwc: NPU 입력 그 자체. uint8 [64, 64, 3] (Mel, 시간, 채널). Mel 행 0 = 고주파.
// 이 버퍼를 다시 복사하지 않고 Run()에 넘긴다.
void features_u8(const float* audio, uint8_t out_hwc[64 * 64 * 3]);
