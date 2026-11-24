#include "features.hpp"

#include <dxrt/dxrt_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSr = 16000;
constexpr int kSec = 16000;
const char* kClasses[] = {
    "whistle_up", "whistle_down", "tongue_click2", "shh", "clap2", "speech", "background",
};

struct Args {
    std::string model = "../models/resnet18_v1/resnet18_v1.dxnn";
    std::string device = "hw:2,0";
    std::string wav;
    std::vector<std::string> sets;
    std::string fps_json;
    int hop = 8000;
    bool dump = false;
};

void usage() {
    std::cerr
        << "사용: recognize [--model dxnn] [--device hw:2,0] [--hop 0.5] [--wav 파일]\n"
        << "      recognize --fps-json fps.json --set psj폴더 [--set jsh폴더]\n"
        << "  마이크는 48 kHz로 받아 3샘플 평균으로 16 kHz로 맞춘 뒤, 1초 창마다 분류합니다.\n"
        << "  --set 은 클래스 폴더를 features.cpp 로 평가하고, Mel 프레임 생성률을 출력합니다.\n"
        << "  wav는 16 kHz 또는 48 kHz, mono, 16-bit 입니다.\n";
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << name << " 값이 필요합니다\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--model") a.model = need("--model");
        else if (k == "--device") a.device = need("--device");
        else if (k == "--wav") a.wav = need("--wav");
        else if (k == "--set") a.sets.push_back(need("--set"));
        else if (k == "--fps-json") a.fps_json = need("--fps-json");
        else if (k == "--hop") a.hop = std::max(1, static_cast<int>(std::atof(need("--hop")) * kSr));
        else if (k == "--dump") a.dump = true;
        else if (k == "--help") {
            usage();
            std::exit(0);
        } else {
            std::cerr << "알 수 없는 인자: " << k << "\n";
            usage();
            return false;
        }
    }
    for (char c : a.device) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == ':' || c == ',' || c == '_' || c == '.')) {
            std::cerr << "장치 이름이 올바르지 않습니다: " << a.device << "\n";
            return false;
        }
    }
    return true;
}

bool read_wav(const std::string& path, int& rate, std::vector<int16_t>& samples) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char riff[12];
    if (std::fread(riff, 1, 12, f) != 12 || std::memcmp(riff, "RIFF", 4) || std::memcmp(riff + 8, "WAVE", 4)) {
        std::fclose(f);
        return false;
    }
    uint16_t audio_format = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data;
    while (true) {
        char id[4];
        uint32_t size = 0;
        if (std::fread(id, 1, 4, f) != 4) break;
        if (std::fread(&size, 4, 1, f) != 1) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t block = 0;
            uint32_t byte_rate = 0;
            if (size < 16) {
                std::fclose(f);
                return false;
            }
            std::fread(&audio_format, 2, 1, f);
            std::fread(&channels, 2, 1, f);
            std::fread(&sample_rate, 4, 1, f);
            std::fread(&byte_rate, 4, 1, f);
            std::fread(&block, 2, 1, f);
            std::fread(&bits, 2, 1, f);
            if (size > 16) std::fseek(f, static_cast<long>(size - 16), SEEK_CUR);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data.resize(size);
            if (std::fread(data.data(), 1, size, f) != size) {
                std::fclose(f);
                return false;
            }
            break;
        } else {
            std::fseek(f, static_cast<long>(size + (size & 1)), SEEK_CUR);
        }
    }
    std::fclose(f);
    if (audio_format != 1 || channels != 1 || bits != 16 || data.size() < 2) return false;
    rate = static_cast<int>(sample_rate);
    samples.resize(data.size() / 2);
    std::memcpy(samples.data(), data.data(), samples.size() * 2);
    return true;
}

void to_16k(const std::vector<int16_t>& in, int rate, std::vector<float>& out) {
    out.clear();
    if (rate == 16000) {
        out.reserve(in.size());
        for (int16_t s : in) out.push_back(static_cast<float>(s) / 32768.f);
        return;
    }
    if (rate != 48000) throw std::runtime_error("샘플레이트는 16000 또는 48000 이어야 합니다");
    out.reserve(in.size() / 3);
    for (size_t i = 0; i + 2 < in.size(); i += 3) {
        const float m = (static_cast<int>(in[i]) + in[i + 1] + in[i + 2]) / 3.0f;
        out.push_back(m / 32768.f);
    }
}

struct Pred {
    int cls;
    float prob;
};

Pred infer(dxrt::InferenceEngine& engine, const float* audio) {
    uint8_t npu[64 * 64 * 3];
    features_u8(audio, npu);
    auto outs = engine.Run(npu);
    const float* logits = static_cast<const float*>(outs.at(0)->data());
    int best = 0;
    float m = logits[0];
    for (int i = 1; i < 7; ++i) {
        if (logits[i] > m) {
            m = logits[i];
            best = i;
        }
    }
    float sum = 0.f;
    float exps[7];
    for (int i = 0; i < 7; ++i) {
        exps[i] = std::exp(logits[i] - m);
        sum += exps[i];
    }
    return {best, exps[best] / sum};
}

void run_windows(dxrt::InferenceEngine& engine, const std::vector<float>& audio, int hop, bool dump) {
    if (static_cast<int>(audio.size()) < kSec) {
        throw std::runtime_error("1초보다 짧은 오디오입니다");
    }
    int index = 0;
    for (int start = 0; start + kSec <= static_cast<int>(audio.size()); start += hop) {
        if (dump && index == 0) {
            uint8_t npu[64 * 64 * 3];
            features_u8(audio.data() + start, npu);
            FILE* f = std::fopen("feat_u8.bin", "wb");
            if (f) {
                std::fwrite(npu, 1, sizeof(npu), f);
                std::fclose(f);
            }
        }
        Pred p = infer(engine, audio.data() + start);
        const double t = static_cast<double>(start) / kSr;
        std::printf("%6.2fs  %-16s  %.3f\n", t, kClasses[p.cls], p.prob);
        std::fflush(stdout);
        ++index;
    }
}

int class_id(const std::string& name) {
    for (int i = 0; i < 7; ++i) {
        if (name == kClasses[i]) return i;
    }
    return -1;
}

void windows_of(const std::vector<float>& audio, std::vector<int>& starts) {
    starts.clear();
    const int n = static_cast<int>(audio.size());
    if (n <= kSec + kSr / 2) {
        starts.push_back(n >= kSec ? (n - kSec) / 2 : 0);
        return;
    }
    for (int i = 0; i + kSec <= n; i += kSec) starts.push_back(i);
}

void evaluate_set(dxrt::InferenceEngine& engine, const std::string& root) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(root)) throw std::runtime_error("폴더가 없습니다: " + root);
    int cm[7][7]{};
    int file_ok = 0;
    int file_n = 0;
    int windows = 0;
    double feat_sec = 0.0;
    double npu_sec = 0.0;
    uint8_t npu[64 * 64 * 3];

    std::vector<fs::path> dirs;
    for (const auto& ent : fs::directory_iterator(root)) {
        if (ent.is_directory() && class_id(ent.path().filename().string()) >= 0) dirs.push_back(ent.path());
    }
    std::sort(dirs.begin(), dirs.end());

    for (const auto& dir : dirs) {
        const int label = class_id(dir.filename().string());
        std::vector<fs::path> wavs;
        for (const auto& ent : fs::directory_iterator(dir)) {
            if (ent.is_regular_file() && ent.path().extension() == ".wav") wavs.push_back(ent.path());
        }
        std::sort(wavs.begin(), wavs.end());
        for (const auto& path : wavs) {
            int rate = 0;
            std::vector<int16_t> pcm;
            if (!read_wav(path.string(), rate, pcm)) throw std::runtime_error("wav 를 읽지 못했습니다: " + path.string());
            if (rate != kSr) throw std::runtime_error("평가 wav 는 16 kHz 여야 합니다: " + path.string());
            std::vector<float> audio;
            to_16k(pcm, rate, audio);
            std::vector<int> starts;
            windows_of(audio, starts);
            int votes[7]{};
            for (int start : starts) {
                std::vector<float> clip(kSec, 0.f);
                if (static_cast<int>(audio.size()) >= kSec) {
                    std::copy(audio.begin() + start, audio.begin() + start + kSec, clip.begin());
                } else {
                    const int left = (kSec - static_cast<int>(audio.size())) / 2;
                    std::copy(audio.begin(), audio.end(), clip.begin() + left);
                }
                const auto t0 = std::chrono::steady_clock::now();
                features_u8(clip.data(), npu);
                const auto t1 = std::chrono::steady_clock::now();
                auto outs = engine.Run(npu);
                const float* logits = static_cast<const float*>(outs.at(0)->data());
                int pred = 0;
                for (int i = 1; i < 7; ++i) {
                    if (logits[i] > logits[pred]) pred = i;
                }
                const auto t2 = std::chrono::steady_clock::now();
                feat_sec += std::chrono::duration<double>(t1 - t0).count();
                npu_sec += std::chrono::duration<double>(t2 - t1).count();
                cm[label][pred] += 1;
                votes[pred] += 1;
                ++windows;
            }
            int maj = 0;
            for (int i = 1; i < 7; ++i) {
                if (votes[i] > votes[maj]) maj = i;
            }
            if (maj == label) ++file_ok;
            ++file_n;
        }
    }

    int correct = 0;
    for (int i = 0; i < 7; ++i) correct += cm[i][i];
    const double total = feat_sec + npu_sec;
    const fs::path name = fs::path(root).filename();
    std::printf("\n== %s  (features.cpp) ==\n", name.c_str());
    std::printf("창 %d개   파일 %d개\n", windows, file_n);
    std::printf("창 정확도   %d/%d = %.4f\n", correct, windows, windows ? static_cast<double>(correct) / windows : 0.0);
    std::printf("파일 정확도 %d/%d = %.4f\n", file_ok, file_n, file_n ? static_cast<double>(file_ok) / file_n : 0.0);
    std::printf("클래스별 재현율\n");
    for (int i = 0; i < 7; ++i) {
        int row = 0;
        for (int j = 0; j < 7; ++j) row += cm[i][j];
        if (row == 0) std::printf("  %-16s  표본 없음\n", kClasses[i]);
        else std::printf("  %-16s  %d/%d = %.3f\n", kClasses[i], cm[i][i], row, static_cast<double>(cm[i][i]) / row);
    }
    std::printf("혼동행렬 (행=정답, 열=예측)\n");
    std::printf("%16s", "");
    for (int j = 0; j < 7; ++j) std::printf(" %8s", kClasses[j]);
    std::printf("\n");
    for (int i = 0; i < 7; ++i) {
        std::printf("%16s", kClasses[i]);
        for (int j = 0; j < 7; ++j) std::printf(" %8d", cm[i][j]);
        std::printf("\n");
    }
    const double frames = static_cast<double>(windows) * 64.0;
    std::printf("시간   특징 %.3fs   NPU %.3fs   합 %.3fs\n", feat_sec, npu_sec, total);
    if (feat_sec > 0.0) {
        std::printf("Mel 프레임 생성률   %.1f 프레임/s   (창 %.1f개/s, 창당 %.2f ms)\n",
                    frames / feat_sec, windows / feat_sec, 1000.0 * feat_sec / windows);
    }
    if (total > 0.0) {
        std::printf("인식 처리율         %.1f 창/s   (%.2f ms/창, 오디오 %.1f초를 %.2fx 로 처리)\n",
                    windows / total, 1000.0 * total / windows,
                    static_cast<double>(windows), static_cast<double>(windows) / total);
    }
    std::fflush(stdout);
}

struct FpsRow {
    std::string name;
    int windows = 0;
    double prep_s = 0;
    double npu_s = 0;
};

FpsRow bench_fps(dxrt::InferenceEngine& engine, const std::string& root) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(root)) throw std::runtime_error("폴더가 없습니다: " + root);
    std::vector<fs::path> wavs;
    std::vector<fs::path> dirs;
    for (const auto& ent : fs::directory_iterator(root)) {
        if (ent.is_directory() && class_id(ent.path().filename().string()) >= 0) dirs.push_back(ent.path());
    }
    std::sort(dirs.begin(), dirs.end());
    for (const auto& dir : dirs) {
        for (const auto& ent : fs::directory_iterator(dir)) {
            if (ent.is_regular_file() && ent.path().extension() == ".wav") wavs.push_back(ent.path());
        }
    }
    std::sort(wavs.begin(), wavs.end());

    constexpr int kBytes = 64 * 64 * 3;
    std::vector<uint8_t> batch;
    FpsRow row;
    row.name = fs::path(root).filename().string();

    for (const auto& path : wavs) {
        int rate = 0;
        std::vector<int16_t> pcm;
        if (!read_wav(path.string(), rate, pcm)) throw std::runtime_error("wav 를 읽지 못했습니다: " + path.string());
        if (rate != kSr) throw std::runtime_error("평가 wav 는 16 kHz 여야 합니다: " + path.string());
        std::vector<float> audio;
        to_16k(pcm, rate, audio);
        std::vector<int> starts;
        windows_of(audio, starts);
        for (int start : starts) {
            std::vector<float> clip(kSec, 0.f);
            if (static_cast<int>(audio.size()) >= kSec) {
                std::copy(audio.begin() + start, audio.begin() + start + kSec, clip.begin());
            } else {
                const int left = (kSec - static_cast<int>(audio.size())) / 2;
                std::copy(audio.begin(), audio.end(), clip.begin() + left);
            }
            const size_t at = batch.size();
            batch.resize(at + kBytes);
            const auto t0 = std::chrono::steady_clock::now();
            features_u8(clip.data(), batch.data() + at);
            const auto t1 = std::chrono::steady_clock::now();
            row.prep_s += std::chrono::duration<double>(t1 - t0).count();
            ++row.windows;
        }
    }

    if (row.windows > 0) {
        auto warm = engine.Run(batch.data());
        (void)warm;
    }
    for (int i = 0; i < row.windows; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto outs = engine.Run(batch.data() + static_cast<size_t>(i) * kBytes);
        const auto t1 = std::chrono::steady_clock::now();
        if (outs.empty()) throw std::runtime_error("NPU 출력이 없습니다");
        row.npu_s += std::chrono::duration<double>(t1 - t0).count();
    }
    const double prep_fps = row.prep_s > 0 ? row.windows / row.prep_s : 0;
    const double npu_fps = row.npu_s > 0 ? row.windows / row.npu_s : 0;
    std::printf("%s  창 %d  전처리 %.1f FPS (%.2f ms)  NPU %.1f FPS (%.2f ms)\n",
                row.name.c_str(), row.windows, prep_fps, 1000.0 * row.prep_s / std::max(row.windows, 1),
                npu_fps, 1000.0 * row.npu_s / std::max(row.windows, 1));
    return row;
}

void write_fps_json(const std::string& path, const std::vector<FpsRow>& rows);

void run_fps_json(dxrt::InferenceEngine& engine, const std::vector<std::string>& sets, const std::string& path) {
    std::vector<FpsRow> rows;
    for (const auto& set : sets) rows.push_back(bench_fps(engine, set));
    write_fps_json(path, rows);
}

void write_fps_json(const std::string& path, const std::vector<FpsRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("fps json 을 쓰지 못했습니다: " + path);
    std::fputs("{\"sets\":[", f);
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const double prep_fps = r.prep_s > 0 ? r.windows / r.prep_s : 0;
        const double npu_fps = r.npu_s > 0 ? r.windows / r.npu_s : 0;
        std::fprintf(f,
                     "%s{\"name\":\"%s\",\"windows\":%d,\"preprocess_s\":%.6f,\"npu_s\":%.6f,"
                     "\"preprocess_fps\":%.4f,\"npu_fps\":%.4f,\"preprocess_ms\":%.4f,\"npu_ms\":%.4f}",
                     i ? "," : "", r.name.c_str(), r.windows, r.prep_s, r.npu_s, prep_fps, npu_fps,
                     r.windows ? 1000.0 * r.prep_s / r.windows : 0.0,
                     r.windows ? 1000.0 * r.npu_s / r.windows : 0.0);
    }
    std::fputs("]}\n", f);
    std::fclose(f);
}

void run_mic(dxrt::InferenceEngine& engine, const std::string& device, int hop) {
    std::ostringstream cmd;
    cmd << "arecord -D " << device << " -f S16_LE -r 48000 -c 1 -t raw -q";
    FILE* fp = popen(cmd.str().c_str(), "r");
    if (!fp) throw std::runtime_error("arecord 를 시작하지 못했습니다");
    std::cerr << "듣는 중: " << device << "  (Ctrl-C 로 종료)\n";

    std::vector<float> buf;
    buf.reserve(kSec + hop);
    int16_t chunk[4800];
    int pending_n = 0;
    int pending_sum = 0;
    int produced = 0;
    while (true) {
        const size_t n = std::fread(chunk, sizeof(int16_t), 4800, fp);
        if (n == 0) break;
        for (size_t i = 0; i < n; ++i) {
            pending_sum += chunk[i];
            if (++pending_n == 3) {
                buf.push_back((pending_sum / 3.0f) / 32768.f);
                pending_sum = 0;
                pending_n = 0;
            }
        }
        while (static_cast<int>(buf.size()) >= kSec) {
            Pred p = infer(engine, buf.data());
            const double t = static_cast<double>(produced) / kSr;
            std::printf("%6.2fs  %-16s  %.3f\n", t, kClasses[p.cls], p.prob);
            std::fflush(stdout);
            const int drop = std::min(hop, static_cast<int>(buf.size()) - 1);
            buf.erase(buf.begin(), buf.begin() + drop);
            produced += drop;
        }
    }
    pclose(fp);
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;
    try {
        dxrt::InferenceEngine engine(args.model);
        if (!args.fps_json.empty()) {
            if (args.sets.empty()) throw std::runtime_error("--fps-json 에는 --set 폴더가 필요합니다");
            std::vector<float> silence(kSec, 0.f);
            infer(engine, silence.data());
            run_fps_json(engine, args.sets, args.fps_json);
        } else if (!args.sets.empty()) {
            std::vector<float> silence(kSec, 0.f);
            infer(engine, silence.data());
            for (const auto& set : args.sets) evaluate_set(engine, set);
        } else if (!args.wav.empty()) {
            int rate = 0;
            std::vector<int16_t> pcm;
            if (!read_wav(args.wav, rate, pcm)) {
                std::cerr << "wav 를 읽지 못했습니다: " << args.wav << "\n";
                return 1;
            }
            std::vector<float> audio;
            to_16k(pcm, rate, audio);
            run_windows(engine, audio, args.hop, args.dump);
        } else {
            run_mic(engine, args.device, args.hop);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
