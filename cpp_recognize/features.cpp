#include "features.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr int kSr = 16000;
constexpr int kN = 16000;
constexpr int kWin = 400;
constexpr int kNfft = 512;
constexpr int kHop = 250;
constexpr int kFrames = 64;
constexpr int kMels = 64;
constexpr int kBins = kNfft / 2 + 1;
constexpr int kDelta = 4;
constexpr double kDbLo = -90.0;
constexpr double kDbHi = 5.0;
constexpr double kD1 = 6.0;
constexpr double kD2 = 1.5;

double hz_to_mel(double f) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double logstep = std::log(6.4) / 27.0;
    if (f >= min_log_hz) {
        return min_log_hz / f_sp + std::log(std::max(f, 1e-10) / min_log_hz) / logstep;
    }
    return f / f_sp;
}

double mel_to_hz(double m) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double logstep = std::log(6.4) / 27.0;
    const double min_log_mel = min_log_hz / f_sp;
    if (m >= min_log_mel) return min_log_hz * std::exp(logstep * (m - min_log_mel));
    return f_sp * m;
}

struct Tables {
    double window[kNfft]{};
    int mel_begin[kMels]{};
    int mel_count[kMels]{};
    int mel_off[kMels]{};
    double mel_w[kMels * kBins]{};

    Tables() {
        double edges[kMels + 2];
        const double mel_lo = hz_to_mel(50.0);
        const double mel_hi = hz_to_mel(8000.0);
        for (int i = 0; i < kMels + 2; ++i) {
            edges[i] = mel_to_hz(mel_lo + (mel_hi - mel_lo) * i / (kMels + 1));
        }
        double width[kMels + 1];
        for (int i = 0; i < kMels + 1; ++i) width[i] = edges[i + 1] - edges[i];

        int filled = 0;
        for (int m = 0; m < kMels; ++m) {
            const double norm = 2.0 / (edges[m + 2] - edges[m]);
            int begin = -1;
            int end = -1;
            double row[kBins];
            for (int b = 0; b < kBins; ++b) {
                const double hz = (kSr / 2.0) * b / (kBins - 1);
                const double rise = -(edges[m] - hz) / width[m];
                const double fall = (edges[m + 2] - hz) / width[m + 1];
                double w = std::min(rise, fall);
                if (w < 0.0) w = 0.0;
                row[b] = w * norm;
                if (row[b] != 0.0) {
                    if (begin < 0) begin = b;
                    end = b + 1;
                }
            }
            if (begin < 0) begin = end = 0;
            mel_begin[m] = begin;
            mel_count[m] = end - begin;
            mel_off[m] = filled;
            for (int b = begin; b < end; ++b) mel_w[filled++] = row[b];
        }

        const int pad = (kNfft - kWin) / 2;
        for (int n = 0; n < kWin; ++n) {
            window[pad + n] = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / kWin);
        }
    }
};

struct Fft {
    double* in = nullptr;
    fftw_complex* out = nullptr;
    fftw_plan plan = nullptr;

    Fft() {
        in = static_cast<double*>(fftw_malloc(sizeof(double) * kNfft));
        out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * kBins));
        if (!in || !out) throw std::runtime_error("FFTW 버퍼를 만들지 못했습니다");
        plan = fftw_plan_dft_r2c_1d(kNfft, in, out, FFTW_ESTIMATE);
        if (!plan) throw std::runtime_error("FFTW plan을 만들지 못했습니다");
    }
    Fft(const Fft&) = delete;
    Fft& operator=(const Fft&) = delete;
    ~Fft() {
        if (plan) fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);
    }
};

Fft& fft_plan() {
    static Fft fft;
    return fft;
}

const Tables& tables() {
    static const Tables t;
    return t;
}

void log_mel(const float* audio, double db[kMels][kFrames]) {
    const auto& tab = tables();
    auto& fft = fft_plan();
    double padded[kN + kNfft]{};
    for (int i = 0; i < kN; ++i) padded[kNfft / 2 + i] = audio[i];

    for (int f = 0; f < kFrames; ++f) {
        const int base = f * kHop;
        for (int n = 0; n < kNfft; ++n) fft.in[n] = padded[base + n] * tab.window[n];
        fftw_execute(fft.plan);
        double power[kBins];
        for (int b = 0; b < kBins; ++b) {
            const double re = fft.out[b][0];
            const double im = fft.out[b][1];
            power[b] = re * re + im * im;
        }
        for (int m = 0; m < kMels; ++m) {
            const double* w = tab.mel_w + tab.mel_off[m];
            const int begin = tab.mel_begin[m];
            const int count = tab.mel_count[m];
            double acc = 0.0;
            for (int k = 0; k < count; ++k) acc += power[begin + k] * w[k];
            db[m][f] = 10.0 * std::log10(acc + 1e-10);
        }
    }
}

void delta(const double in[kMels][kFrames], double out[kMels][kFrames]) {
    const double denom = 2.0 * (1 + 4 + 9 + 16);
    for (int m = 0; m < kMels; ++m) {
        double p[kFrames + 2 * kDelta];
        for (int t = 0; t < kFrames; ++t) p[kDelta + t] = in[m][t];
        for (int t = 0; t < kDelta; ++t) {
            p[t] = in[m][0];
            p[kDelta + kFrames + t] = in[m][kFrames - 1];
        }
        for (int t = 0; t < kFrames; ++t) {
            double acc = 0.0;
            for (int i = 1; i <= kDelta; ++i) {
                acc += i * (p[kDelta + t + i] - p[kDelta + t - i]);
            }
            out[m][t] = acc / denom;
        }
    }
}

uint8_t to_u8(double v, double lo, double hi) {
    double x = (v - lo) / (hi - lo);
    if (x < 0.0) x = 0.0;
    if (x > 1.0) x = 1.0;
    return static_cast<uint8_t>(std::llround(x * 255.0));
}

}  // namespace

void features_u8(const float* audio, uint8_t out_hwc[64 * 64 * 3]) {
    double db[kMels][kFrames];
    double d1[kMels][kFrames];
    double d2[kMels][kFrames];
    log_mel(audio, db);
    delta(db, d1);
    delta(d1, d2);
    for (int mel = 0; mel < kMels; ++mel) {
        const int row = kMels - 1 - mel;
        for (int t = 0; t < kFrames; ++t) {
            uint8_t* px = out_hwc + (row * kFrames + t) * 3;
            px[0] = to_u8(db[mel][t], kDbLo, kDbHi);
            px[1] = to_u8(d1[mel][t], -kD1, kD1);
            px[2] = to_u8(d2[mel][t], -kD2, kD2);
        }
    }
}
