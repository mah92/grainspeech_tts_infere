#include "normalize.h"
#include "grain_phones.h"
#include "language_detector/language_detector.h"

#include <onnxruntime_cxx_api.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#include <android/log.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <locale>
#include <codecvt>
#include <complex>

// POSIX sockets + process control
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>

// ============================================================================
// Global socket path
// ============================================================================
#if defined(__ANDROID__)
#include <android/log.h>
#define GRAIN_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "SalamNativeGrain", __VA_ARGS__)
#else
#define GRAIN_LOGI(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while (0)
#endif

static const char* SOCKET_PATH = "/tmp/grainspeech_infer.sock";   // never share the Matcha daemon socket

// ============================================================================
// WAV Writer: 16-bit PCM, mono
// ============================================================================
static void write_wav(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Cannot open output file: " << path << std::endl;
        return;
    }

    int num_samples = static_cast<int>(audio.size());
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int data_size = num_samples * block_align;
    int chunk_size = 36 + data_size;

    // RIFF header
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    out.write("WAVE", 4);

    // fmt subchunk
    out.write("fmt ", 4);
    int fmt_size = 16;
    short audio_format = 1; // PCM
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&num_channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data subchunk
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), 4);

    // Convert float [-1,1] to int16
    for (float sample : audio) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
        int16_t int_sample = static_cast<int16_t>(sample * 32767.0f);
        out.write(reinterpret_cast<const char*>(&int_sample), 2);
    }

    out.close();
}

// ============================================================================
// Token map reader
// ============================================================================
static int read_token_map(const std::string& filepath,
                          std::map<std::string, int>& token_to_id,
                          std::map<int, std::string>& id_to_token) {
    // The GrainSpeech symbol table: one token per line, the line number IS the id
    // (generated from grainspeech/text/symbols.py by export_grainspeech_onnx.py).
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open symbols file: " << filepath << std::endl;
        return -1;
    }
    std::string line;
    int id = 0;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) { id++; continue; }
        token_to_id[line] = id;          // keys keep the leading '@' as written in symbols.py
        id_to_token[id] = line;
        id++;
    }
    std::cout << "Loaded " << token_to_id.size() << " symbols from " << filepath << std::endl;
    return 0;
}

// ============================================================================
// Convert IPA string to token IDs
// ============================================================================
// ============================================================================
// Phone tokens (already in the model's convention) -> symbol ids
// ============================================================================
// ---------------------------------------------------------------------------
// Phone normalization for the model inventory (Ali 2026-10-02).
//
// Every word is phonemized by espeak-ng (there is no lexicon any more), and espeak
// emits phones this model's alphabet does not carry: the near-close central
// vowel ᵻ, the tap ɾ, ɐ/ɒ, a bare long ɑː, and diphthongs that espeak splits
// into two tokens ("ˌe" + "ɪ" for eɪ). Those phones were silently SKIPPED by
// phones_to_ids, which truncated words on-device: TalkBack's «double tap to
// activate» reached the ear as «activ» because ᵻ and ˌe vanished.
//
// Each unknown phone is mapped onto the nearest symbol the model actually
// carries (this symbol table has 69 English / 74 Persian phone symbols).
// Anything still unknown is left in place and reported exactly as before.
// ---------------------------------------------------------------------------
static std::vector<std::string> normalize_phones_for_inventory(
        const std::vector<std::string>& in,
        const std::map<std::string, int>& token_to_id) {
    // NOTE: symbol-table keys carry the leading '@' (phones_to_ids adds it), so
    // every lookup here must add it too — forgetting this made the whole
    // normalization a silent no-op on the first attempt.
    auto has = [&](const std::string& k) {
        std::string key = (!k.empty() && k[0] == '@') ? k : ("@" + k);
        return token_to_id.find(key) != token_to_id.end();
    };
    auto split_prefix = [](const std::string& p, std::string& prefix, std::string& core) {
        size_t c = p.find(':');
        if (c == std::string::npos) { prefix.clear(); core = p; }
        else { prefix = p.substr(0, c + 1); core = p.substr(c + 1); }
    };
    // strip leading primary ˈ (U+02C8 = CB 88) / secondary ˌ (U+02CC = CB 8C) marks
    auto split_stress = [](const std::string& core, std::string& stress, std::string& rest) {
        stress.clear();
        rest = core;
        for (;;) {
            if (rest.size() >= 2
                && static_cast<unsigned char>(rest[0]) == 0xCB
                && (static_cast<unsigned char>(rest[1]) == 0x88
                    || static_cast<unsigned char>(rest[1]) == 0x8C)) {
                stress += rest.substr(0, 2);
                rest.erase(0, 2);
            } else {
                break;
            }
        }
    };

    const std::string V_A = "a", V_E = "e", V_O = "o", V_I = "i", V_U = "u";
    const std::string V_OPEN_O = "\xC9\x94";        // ɔ
    const std::string P_I = "\xC9\xAA";             // ɪ
    const std::string P_U = "\xCA\x8A";             // ʊ
    const std::string P_SCHWA = "\xC9\x99";         // ə
    const std::string P_LEN = "\xCB\x90";           // ː
    const std::string S_PRIMARY = "\xCB\x88";       // ˈ
    const std::string S_SECONDARY = "\xCB\x8C";     // ˌ

    std::vector<std::string> out;
    for (size_t i = 0; i < in.size(); ++i) {
        std::string prefix, core;
        split_prefix(in[i], prefix, core);
        std::string stress, rest;
        split_stress(core, stress, rest);

        // 1) already a symbol the model knows
        if (has(in[i])) { out.push_back(in[i]); continue; }

        // 2) an espeak-split diphthong: ("ˈa","ɪ") -> "ˈaɪ", ("ˌe","ɪ") -> "ˌeɪ"
        bool is_vowel_letter = (rest == V_A || rest == V_E || rest == V_O
                                || rest == V_I || rest == V_U || rest == V_OPEN_O);
        if (is_vowel_letter && i + 1 < in.size()) {
            std::string np, ncore;
            split_prefix(in[i + 1], np, ncore);
            std::string nstress, nrest;
            split_stress(ncore, nstress, nrest);
            if (np == prefix && (nstress.empty() || nstress == stress)
                && (nrest == P_I || nrest == P_U)) {
                std::string merged = prefix + stress + rest + nrest;
                if (has(merged)) { out.push_back(merged); ++i; continue; }
                std::string bare = prefix + rest + nrest;
                if (has(bare)) { out.push_back(bare); ++i; continue; }
            }
        }

        // 3) near-miss substitutions (stress-preserving form first)
        static const char* kSubs[][2] = {
            {"\xE1\xB5\xBB", "\xC9\xAA"},                    // ᵻ  -> ɪ
            {"\xC9\xA8",     "\xC9\xAA"},                    // ɨ  -> ɪ
            {"\xC9\xBE",     "\xC9\xB9"},                    // ɾ  -> ɹ
            {"r",            "\xC9\xB9"},                    // r  -> ɹ
            {"\xC9\x90",     "\xC9\x99"},                    // ɐ  -> ə
            {"\xC9\x92",     "\xC9\x94"},                    // ɒ  -> ɔ
            {"\xCA\x8C",     "\xC9\x99"},                    // ʌ  -> ə
            {"\xC9\x9C",     "\xC9\x9C\xCB\x90"},            // ɜ  -> ɜː
            {"\xC9\x9A",     "\xC9\x99"},                    // ɚ  -> ə
            {"\xC9\x9D",     "\xC9\x9C\xCB\x90"},            // ɝ  -> ɜː
        };
        if (rest.size() <= 3) {
            bool done = false;
            for (const auto& sub : kSubs) {
                if (rest == sub[0]) {
                    std::string cand = prefix + stress + sub[1];
                    if (has(cand)) { out.push_back(cand); done = true; break; }
                    std::string cand2 = prefix + sub[1];
                    if (has(cand2)) { out.push_back(cand2); done = true; break; }
                }
            }
            if (done) continue;
        }

        // 4) drop a trailing length mark (ɑː -> ɑ where only the short form exists)
        if (rest.size() > 2 && rest.compare(rest.size() - 2, 2, P_LEN) == 0) {
            std::string stripped = rest.substr(0, rest.size() - 2);
            std::string cand = prefix + stress + stripped;
            if (has(cand)) { out.push_back(cand); continue; }
            std::string cand2 = prefix + stripped;
            if (has(cand2)) { out.push_back(cand2); continue; }
            std::string cand3 = prefix + stripped + P_LEN;
            if (has(cand3)) { out.push_back(cand3); continue; }
        }

        // 5) a secondary-stressed vowel that only exists unstressed (ˌo -> o)
        if (stress == S_SECONDARY) {
            std::string bare = prefix + rest;
            if (has(bare)) { out.push_back(bare); continue; }
        }

        // 6) split a vowel+schwa sequence (ɪə -> ɪ + ə) when both halves exist
        if (rest.size() > 2 && rest.compare(rest.size() - 2, 2, P_SCHWA) == 0) {
            std::string head = rest.substr(0, rest.size() - 2);
            if (has(prefix + head) && has(prefix + P_SCHWA)) {
                out.push_back(prefix + stress + head);
                out.push_back(prefix + P_SCHWA);
                continue;
            }
        }

        // 7) STRIP_DIACRITICS: drop phonetic diacritics (palatalisation ʲ,
        //    aspiration ʰ, nasalisation ̃, syllabic ̩ …) and retry — espeak emits
        //    them for foreign words/URLs and the model's alphabet has none of them.
        {
            std::string stripped;
            for (size_t k = 0; k < rest.size();) {
                size_t len = 1;
                unsigned char c0 = static_cast<unsigned char>(rest[k]);
                if ((c0 & 0xE0) == 0xC0) len = 2;
                else if ((c0 & 0xF0) == 0xE0) len = 3;
                else if ((c0 & 0xF8) == 0xF0) len = 4;
                if (k + len > rest.size()) len = 1;
                std::string cp = rest.substr(k, len);
                k += len;
                static const char* kDiacritics[] = {
                    "\xCA\xB0",  // ʰ
                    "\xCA\xB2",  // ʲ
                    "\xCA\xB7",  // ʷ
                    "\xCB\xA4",  // ˤ
                    "\xCB\xA0",  // ˠ
                    "\xCB\x80",  // ˀ
                    "\xCC\xA9",  // ̩
                    "\xCC\xAF",  // ̯
                    "\xCC\x83",  // ̃
                    "\xCC\xA5",  // ̥
                    "\xCA\xB1",  // ʱ
                    "\xCA\xA1"   // ˡ
                };
                bool drop = false;
                for (const char* d : kDiacritics) {
                    if (cp == d) { drop = true; break; }
                }
                if (!drop) stripped += cp;
            }
            if (!stripped.empty() && stripped != rest) {
                std::string cand = prefix + stress + stripped;
                if (has(cand)) { out.push_back(cand); continue; }
                std::string cand2 = prefix + stripped;
                if (has(cand2)) { out.push_back(cand2); continue; }
            }
        }

        // 8) still unknown — keep it; phones_to_ids reports and skips it
        out.push_back(in[i]);
        (void)S_PRIMARY;
    }
    return out;
}

static std::vector<int64_t> phones_to_ids(const std::vector<std::string>& phones,
                                          const std::map<std::string, int>& token_to_id,
                                          int& missing) {
    std::vector<int64_t> ids;
    missing = 0;
    for (const auto& p : phones) {
        std::string key = (!p.empty() && p[0] == '@') ? p : ("@" + p);
        auto it = token_to_id.find(key);
        if (it != token_to_id.end()) {
            ids.push_back(static_cast<int64_t>(it->second));
        } else {
            ++missing;
            std::cerr << "Warning: phone '" << p << "' not in the symbol table — skipped" << std::endl;
        }
    }
    return ids;
}

// ============================================================================
// FFT / ISTFT — for reconstructing waveform from Vocos mag/x/y (ISTFT) output.
// k2-fsa vocos-22khz-univ.onnx outputs (mag, x, y) instead of a direct waveform.
// ============================================================================
static void fft(std::vector<std::complex<float>>& a, bool invert) {
    int n = (int)a.size();
    // bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (invert ? 1.0 : -1.0);
        std::complex<float> wlen((float)std::cos(ang), (float)std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; j++) {
                std::complex<float> u = a[i + j];
                std::complex<float> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (auto& x : a) x /= (float)n;
    }
}

// Reconstruct waveform from Vocos ISTFT head output: S = mag * (x + i*y),
// then overlap-add inverse STFT (hann window, center=True, hop=256, n_fft=1024).
static std::vector<float> vocos_istft(const float* mag, const float* x, const float* y,
                                      int n_fft, int hop, int frames) {
    int n_bins = n_fft / 2 + 1;
    int output_len = hop * (frames - 1);

    std::vector<float> out((size_t)output_len, 0.0f);
    std::vector<float> win_sum((size_t)output_len, 0.0f);

    // hann window
    std::vector<float> window((size_t)n_fft);
    for (int i = 0; i < n_fft; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n_fft - 1)));
    }

    for (int f = 0; f < frames; f++) {
        std::vector<std::complex<float>> spec((size_t)n_fft);
        // ONNX mag/x/y are bin-major: [bin][frame] → element = bin * frames + frame
        for (int k = 0; k < n_bins; k++) {
            size_t idx = (size_t)k * frames + f;
            spec[k] = { mag[idx] * x[idx], mag[idx] * y[idx] };
        }
        // conjugate symmetry for the upper half
        for (int k = 1; k < n_fft / 2; k++) {
            spec[n_fft - k] = std::conj(spec[k]);
        }

        fft(spec, true);  // inverse FFT

        // overlap-add: frame f at offset f*hop, then trim n_fft/2 both sides
        int start = f * hop - n_fft / 2;
        for (int i = 0; i < n_fft; i++) {
            int pos = start + i;
            if (pos >= 0 && pos < output_len) {
                out[pos] += spec[i].real() * window[i];
                win_sum[pos] += window[i] * window[i];
            }
        }
    }

    for (int i = 0; i < output_len; i++) {
        if (win_sum[i] > 1e-6f) out[i] /= win_sum[i];
    }

    return out;
}

// ============================================================================
// JSON helpers (minimal — no library dependency)
// ============================================================================
static std::string json_esc(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
        if (c == '\r') { out += "\\r"; continue; }
        out += c;
    }
    return out;
}

static std::string json_str(const std::string& s) {
    return "\"" + json_esc(s) + "\"";
}

// Simple JSON value extractor: find "key":"value" or "key":number
static std::string json_get_str(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        pos++;
        std::string val;
        while (pos < json.size()) {
            if (json[pos] == '\\') {
                pos++;
                if (pos < json.size()) {
                    char n = json[pos];
                    if (n == 'n') val += '\n';
                    else if (n == 'r') val += '\r';
                    else val += n;
                }
            } else if (json[pos] == '"') {
                break;
            } else {
                val += json[pos];
            }
            pos++;
        }
        return val;
    }

    if (json[pos] == '-' || (json[pos] >= '0' && json[pos] <= '9')) {
        size_t end = pos;
        while (end < json.size() && (json[end] == '-' || json[end] == '.' ||
               (json[end] >= '0' && json[end] <= '9'))) end++;
        return json.substr(pos, end - pos);
    }

    return "";
}

// ============================================================================
// Config and result types
// ============================================================================
struct SynthConfig {
    std::string acoustic_model;   // GrainSpeech phones -> mel (exported ONNX)
    std::string symbols_file;     // symbol table, one token per line (line number = id)
    std::string vocoder_model;
    std::string tokens_file;
    std::string espeak_data;
    std::string ezafe_onnx;
    std::string ezafe_spiece;
    std::string hazm_words;
    std::string hazm_verbs;
    std::string hazm_stopwords;
    std::string homograph_data;
    std::string shakkelha_onnx;
    std::string mel_transform;
    std::string main_lang_str = "FA";
    float temperature = 0.667f;
    float speed = 1.5f;
    int sample_rate = 16000;
    bool use_gpu = false;
    bool debug = false;
};

struct SynthResult {
    bool ok = false;
    std::string error;
    std::string output_path;
    double duration_secs = 0;
    double norm_ms = 0;
    double matcha_ms = 0;
    double vocos_ms = 0;
    double total_ms = 0;
};

// ============================================================================
// Global state (loaded once in daemon, once per run in direct mode)
// ============================================================================
static std::unique_ptr<Ort::Env> g_env;
static Ort::SessionOptions g_session_opts;
static Ort::MemoryInfo g_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
static bool g_models_loaded = false;

// Optional 80x80 mel-style transform (maps matcha log-mel mags to the vocoder's
// training mel style, e.g. slaney -> htk). Loaded once + reused across requests.
static std::vector<float> g_mel_transform;
static bool g_mel_transform_loaded = false;

static std::map<std::string, int> g_token_to_id;
static std::map<int, std::string> g_id_to_token;
static NormalizeConfig g_norm_config;
static Language g_mainlang;
static std::string g_lang_tag = "fa";

static std::unique_ptr<Ort::Session> g_acoustic_session;
static std::unique_ptr<Ort::Session> g_vocoder_session;

static bool load_all_models(const SynthConfig& cfg) {
    if (g_models_loaded) return true;

    // Model paths are passed externally — validate before loading.
    if (cfg.acoustic_model.empty() || cfg.vocoder_model.empty() || cfg.symbols_file.empty()) {
        std::cerr << "Error: missing model paths. Pass --model, --vocoder, --symbols (and optionally --espeak-data)." << std::endl;
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    g_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tts_infer"));
    g_session_opts.SetIntraOpNumThreads(4);
#if defined(__ANDROID__)
    // ANDROID (SalamTTS, ORT 1.18): ANY graph optimization >= BASIC makes
    // libonnxruntime SIGBUS (BUS_ADRALN) during session creation for this model
    // family on ARMv7 — the exact failure the app's matcha / shakkelha / ezafe
    // sessions had to be fixed for (ADR-014 / ADR-031). The desktop build keeps
    // ORT_ENABLE_ALL (it is validated there and is faster).
    //
    // 2026-10-03 (TalkBack investigation): the engine now returns nonsense on the
    // device — the same text that yields 12 480 samples on the desktop comes back as
    // 1 sample, or as millions of samples (2 170 555 / 3 858 763 / 5 551 056 recur
    // for different inputs), while the identical model and code are correct on x86.
    // That points at the ORT session configuration (optimization level and the
    // disabled CPU memory arena / memory pattern) rather than at the model, so both
    // are runtime-selectable:
    //
    //   adb shell setprop debug.grain.opt   0|1|2|3   (0=disable all, 3=enable all)
    //   adb shell setprop debug.grain.arena 0|1       (1 = keep ORT's CPU arena)
    //
    // Defaults keep today's shipping behaviour (opt 0, arena disabled); a process
    // restart (force-stop) is required because the session is built at init.
    {
        char buf[PROP_VALUE_MAX] = {0};
        int lvl = 0;
        if (__system_property_get("debug.grain.opt", buf) > 0) lvl = std::atoi(buf);
        GraphOptimizationLevel gl = GraphOptimizationLevel::ORT_DISABLE_ALL;
        switch (lvl) {
            case 1: gl = GraphOptimizationLevel::ORT_ENABLE_BASIC; break;
            case 2: gl = GraphOptimizationLevel::ORT_ENABLE_EXTENDED; break;
            case 3: gl = GraphOptimizationLevel::ORT_ENABLE_ALL; break;
            default: gl = GraphOptimizationLevel::ORT_DISABLE_ALL; break;
        }
        g_session_opts.SetGraphOptimizationLevel(gl);
        char ab[PROP_VALUE_MAX] = {0};
        // DEFAULT IS TO KEEP THE ARENA (device finding, 2026-10-03): with the CPU arena and the
        // memory pattern DISABLED (the old memory-driven default) every synthesis after the first
        // returned 1 sample even when the vocoder session was rebuilt — while the identical model
        // and code are correct on x86. The run that produced 19 200 / 20 992 / 21 760 samples had
        // cpu-arena=1, so the defaults now follow ORT (arena and memory pattern ON) and the
        // memory-bounded behaviour is only used when explicitly requested with
        // debug.grain.arena 0.
        int arena = 1;
        if (__system_property_get("debug.grain.arena", ab) > 0) arena = std::atoi(ab);
        if (!arena) {
            // Keep resident memory bounded: without these, ORT's CPU arena holds all
            // intermediate activations and never returns memory to the OS (~4 GB RSS).
            g_session_opts.DisableCpuMemArena();
            g_session_opts.DisableMemPattern();
        }
        __android_log_print(ANDROID_LOG_INFO, "SalamNativeGrain",
                            "ORT session: opt-level=%d cpu-arena=%d", lvl, arena);
    }
#else
    g_session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    g_session_opts.DisableCpuMemArena();
    g_session_opts.DisableMemPattern();
#endif

    if (read_token_map(cfg.symbols_file, g_token_to_id, g_id_to_token) != 0) {
        return false;
    }

    g_lang_tag = (cfg.main_lang_str == "EN" || cfg.main_lang_str == "en") ? "en" : "fa";
    std::cout << "Loading GrainSpeech acoustic model: " << cfg.acoustic_model << std::endl;
    try {
        g_acoustic_session.reset(new Ort::Session(*g_env, cfg.acoustic_model.c_str(), g_session_opts));
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load the acoustic model: " << e.what() << std::endl;
        return false;
    }

    std::cout << "Loading Vocoder model: " << cfg.vocoder_model << std::endl;
    try {
        g_vocoder_session.reset(new Ort::Session(*g_env, cfg.vocoder_model.c_str(), g_session_opts));
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load Vocoder model: " << e.what() << std::endl;
        return false;
    }

    g_norm_config.espeak_data_path = cfg.espeak_data;
    g_norm_config.shakkelha_onnx   = cfg.shakkelha_onnx;
    g_norm_config.ezafe_model_onnx = cfg.ezafe_onnx;
    g_norm_config.ezafe_model_spiece = cfg.ezafe_spiece;
    g_norm_config.hazm_words       = cfg.hazm_words;
    g_norm_config.hazm_verbs       = cfg.hazm_verbs;
    g_norm_config.hazm_stopwords   = cfg.hazm_stopwords;
    g_norm_config.homograph_data   = cfg.homograph_data;

    g_mainlang = LanguageDetector::string_to_language(cfg.main_lang_str);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "All models loaded in " << load_secs << "s" << std::endl;

    g_models_loaded = true;
    return true;
}

// ============================================================================
// Speed control
// ============================================================================
// GrainSpeech has NO length-scale input: the durations come from its own duration head, so a
// speed factor cannot be fed to the acoustic model (unlike Matcha's `scales`). It is applied to
// the mel along the TIME axis instead — the same recipe the Python pipeline uses
// (synthesize_fa16k.py: stretch_time, ratio = 1/speed), i.e. linear interpolation of the mel
// rows. Frames become `1/speed` times as many, so speed 1.6 speaks 1.6x faster and the pitch is
// unchanged (this is why it is an interpolation and NOT a resample of the waveform).
static std::vector<float> stretch_mel_time(const std::vector<float>& mel, int64_t n_mels,
                                           int64_t frames, float speed) {
    if (speed <= 0.0f || std::fabs(speed - 1.0f) < 0.01f || frames < 2) return mel;
    int64_t new_frames = static_cast<int64_t>(std::llround(static_cast<double>(frames) / speed));
    if (new_frames < 2) new_frames = 2;
    std::vector<float> out(static_cast<size_t>(n_mels) * static_cast<size_t>(new_frames), 0.0f);
    for (int64_t t = 0; t < new_frames; ++t) {
        double src = static_cast<double>(t) * static_cast<double>(speed);
        if (src > static_cast<double>(frames - 1)) src = static_cast<double>(frames - 1);
        int64_t i0 = static_cast<int64_t>(src);
        int64_t i1 = (i0 + 1 < frames) ? (i0 + 1) : i0;
        float frac = static_cast<float>(src - static_cast<double>(i0));
        for (int64_t m = 0; m < n_mels; ++m) {
            float a = mel[static_cast<size_t>(m) * frames + i0];
            float b = mel[static_cast<size_t>(m) * frames + i1];
            out[static_cast<size_t>(m) * new_frames + t] = a + (b - a) * frac;
        }
    }
    return out;
}

// ============================================================================
// Core synthesis
// ============================================================================
// forward declarations: the definitions live further down, next to the segmenting front-end
static void fadeEdges(std::vector<float>& v, int sample_rate);
static void trimOnset(std::vector<float>& v, int sample_rate);

static double envSeconds(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    const double d = std::atof(v);
    return (d >= 0.0) ? d : fallback;
}

// Raised-cosine ramp; length from GRAIN_FADE_MS (default 5). A 5 ms LINEAR ramp was not enough
// for the onset thump (measured: the 15 ms after a pause peaks at 2000-3900 while ordinary
// speech peaks at 400-800), so the length and shape are both tunable here.
static void fadeEdges(std::vector<float>& v, int sample_rate) {
    const size_t n = v.size();
    if (sample_rate <= 0) return;
    const size_t f = std::min<size_t>((size_t)(envSeconds("GRAIN_FADE_MS", 30.0) * 0.001 * sample_rate), n / 2);
    if (f == 0) return;
    for (size_t k = 0; k < f; ++k) {
        const float g = 0.5f * (1.0f - std::cos(3.14159265f * (float)k / (float)f));  // 0 -> 1
        v[k] *= g;                                    // cosine ramp in
        v[n - 1 - k] *= g;                            // cosine ramp out
    }
}

// GRAIN_TRIM_ONSET_MS > 0 drops that much audio from the start of every piece: the model's first
// phones carry a sharp attack transient which is exactly what the ear reports as a «تیک».
static void trimOnset(std::vector<float>& v, int sample_rate) {
    const double ms = envSeconds("GRAIN_TRIM_ONSET_MS", 20.0);
    if (ms <= 0.0 || v.empty()) return;
    if (sample_rate <= 0) return;                       // never guess a rate
    const size_t wanted = (size_t)(ms * 0.001 * (double)sample_rate);
    // Safety net (2026-10-03 device finding): a corrupted sample_rate once made this erase the
    // whole piece — the engine then returned 1 sample, the screen reader heard nothing and the
    // app was blamed. A trim of a quarter of the audio or more is never legitimate: skip it.
    if (wanted * 4 >= v.size()) {
        GRAIN_LOGI("trimOnset skipped: wanted=%d samples of %d (sample_rate=%d)",
                   (int)wanted, (int)v.size(), sample_rate);
        return;
    }
    const size_t cut = std::min<size_t>(wanted, v.size() - 1);
    v.erase(v.begin(), v.begin() + cut);
}

static SynthResult synthesize_one(const SynthConfig& cfg, const std::string& text,
                                  const std::string& output_path,
                                  std::vector<float>* audio_out = nullptr,
                                  bool quiet = false) {
    SynthResult result;
    auto t_total_start = std::chrono::high_resolution_clock::now();

    // the language can change per request (daemon serves FA and EN)
    g_lang_tag = (cfg.main_lang_str == "EN" || cfg.main_lang_str == "en") ? "en" : "fa";
    g_mainlang = LanguageDetector::string_to_language(cfg.main_lang_str);
    std::string normalized_text;
    std::string ipa_text;

    if (cfg.debug) std::cout << "Normalizing text: " << text << std::endl;
    auto t_norm_start = std::chrono::high_resolution_clock::now();
    normalizeString(g_mainlang, 1 /* IPA mode */, text,
                    normalized_text, ipa_text, g_norm_config);
    auto t_norm_end = std::chrono::high_resolution_clock::now();

    ipa_text.erase(std::remove(ipa_text.begin(), ipa_text.end(), '\n'), ipa_text.end());

    if (cfg.debug) {
        std::cout << "IPA phonemes: " << ipa_text << std::endl;
        std::cout << "Normalized:   " << normalized_text << std::endl;
    }

    result.norm_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_norm_start).count();
    fprintf(stderr, "[TIMING] NormalizeText : %.1f ms\n", result.norm_ms);

    // --- GrainSpeech acoustic model: NormalizeText phones -> mel ---
    if (cfg.debug) std::cout << "\n=== GrainSpeech acoustic model ===" << std::endl;

    std::string phone_str = grain::toModelPhones(normalized_text, ipa_text, g_lang_tag);
    std::vector<std::string> phones;
    {
        std::istringstream ss(phone_str);
        std::string tk;
        while (ss >> tk) phones.push_back(tk);
    }
    // map espeak-only phones onto symbols the model carries, so a word is not
    // silently truncated by a missing symbol (see the function above)
    phones = normalize_phones_for_inventory(phones, g_token_to_id);
    if (cfg.debug) {
        std::string np;
        for (size_t k = 0; k < phones.size(); ++k) { if (k) np += " "; np += phones[k]; }
        std::cout << "Phones (normalized):" << np << std::endl;
    }
    int missing = 0;
    std::vector<int64_t> token_ids = phones_to_ids(phones, g_token_to_id, missing);
    if (cfg.debug) {
        std::cout << "Phones (" << phones.size() << ", " << missing << " unknown):";
        for (const auto& p : phones) std::cout << " " << p;
        std::cout << std::endl;
    }
    if (token_ids.empty()) {
        std::cerr << "Error: no phone tokens produced for this text" << std::endl;
        return result;
    }
    GRAIN_LOGI("model input: text=%d bytes phones=%d token_ids=%d unknown=%d",
               (int)text.size(), (int)phones.size(), (int)token_ids.size(), missing);
    int64_t seq_len = static_cast<int64_t>(token_ids.size());

    std::vector<int64_t> x_shape = {1, seq_len};
    Ort::Value x_tensor = Ort::Value::CreateTensor<int64_t>(
        g_mem_info, token_ids.data(), token_ids.size(), x_shape.data(), x_shape.size());

    const char* ac_input_names[] = {"phoneme"};
    static const char* ac_output_names[] = {"mel", "mel_len"};

    auto t_ac_start = std::chrono::high_resolution_clock::now();
    auto ac_outputs = g_acoustic_session->Run(
        Ort::RunOptions{nullptr}, ac_input_names, &x_tensor, 1, ac_output_names, 2);
    auto t_ac_end = std::chrono::high_resolution_clock::now();
    result.matcha_ms = std::chrono::duration<double, std::milli>(t_ac_end - t_ac_start).count();
    fprintf(stderr, "[TIMING] acoustic model  : %.1f ms\n", result.matcha_ms);

    auto mel_info = ac_outputs[0].GetTensorTypeAndShapeInfo();
    auto mel_shape_raw = mel_info.GetShape();               // (1, T, 80)
    if (mel_shape_raw.size() != 3) {
        std::cerr << "Error: unexpected mel shape from the acoustic model" << std::endl;
        return result;
    }
    int64_t mel_frames = mel_shape_raw[1];
    int64_t n_mels = mel_shape_raw[2];
    GRAIN_LOGI("acoustic output: mel frames=%d n_mels=%d", (int)mel_frames, (int)n_mels);
    const float* mel_raw = ac_outputs[0].GetTensorData<float>();
    if (cfg.debug)
        std::cout << "Mel: [" << mel_shape_raw[0] << ", " << mel_frames << ", " << n_mels << "]" << std::endl;

    // transpose (1, T, 80) -> (80, T), the layout the vocoder expects
    static std::vector<float> mel_t;
    mel_t.assign(static_cast<size_t>(n_mels) * mel_frames, 0.0f);
    for (int64_t i = 0; i < n_mels; ++i)
        for (int64_t t2 = 0; t2 < mel_frames; ++t2)
            mel_t[i * mel_frames + t2] = mel_raw[t2 * n_mels + i];
    float* mel_data = mel_t.data();

    // Speed: resample the mel along time (see stretch_mel_time above). Done BEFORE the vocoder
    // and without touching cfg.speed's meaning for the rest of the pipeline.
    static std::vector<float> mel_speed;
    if (cfg.speed > 0.0f && std::fabs(cfg.speed - 1.0f) >= 0.01f) {
        mel_speed = stretch_mel_time(mel_t, n_mels, mel_frames, cfg.speed);
        mel_frames = static_cast<int64_t>(mel_speed.size() / static_cast<size_t>(n_mels));
        mel_data = mel_speed.data();
        if (cfg.debug) std::cout << "Speed " << cfg.speed << " -> mel frames " << mel_frames << std::endl;
    }

    // --- Vocos vocoder inference (k2-fsa vocos-22khz-univ: mels → mag/x/y) ---
    if (cfg.debug) std::cout << "\n=== Vocoder Inference ===" << std::endl;

    std::vector<int64_t> mel_input_shape = {1, 80, mel_frames};

    float* vocoder_mel_data = mel_data;
    std::vector<float> mel_transformed;
    if (!cfg.mel_transform.empty()) {
        if (!g_mel_transform_loaded) {
            std::ifstream tf(cfg.mel_transform);
            if (!tf) {
                result.error = "Cannot open --mel-transform file: " + cfg.mel_transform;
                return result;
            }
            float v;
            g_mel_transform.clear();
            while (tf >> v) g_mel_transform.push_back(v);
            if (g_mel_transform.size() != static_cast<size_t>(80 * 80)) {
                result.error = "--mel-transform must contain 80x80 floats, got " +
                               std::to_string(g_mel_transform.size());
                return result;
            }
            g_mel_transform_loaded = true;
        }
        // mel_data is log-mel (1, 80, T), NCHW layout. Map it to the vocoder's
        // training mel style: mag = exp(mel); htk_mag[j] = sum_i M[j][i]*mag[i];
        // out = log(max(htk_mag, 1e-5)).
        mel_transformed.resize(static_cast<size_t>(80) * mel_frames);
        for (int64_t t = 0; t < mel_frames; t++) {
            float mag[80];
            for (int i = 0; i < 80; i++) mag[i] = std::exp(mel_data[i * mel_frames + t]);
            for (int j = 0; j < 80; j++) {
                float s = 0.0f;
                for (int i = 0; i < 80; i++) s += g_mel_transform[j * 80 + i] * mag[i];
                s = std::fmax(s, 1e-5f);
                mel_transformed[j * mel_frames + t] = std::log(s);
            }
        }
        vocoder_mel_data = mel_transformed.data();
        if (cfg.debug) std::cout << "Applied --mel-transform (80x80) to matcha mels" << std::endl;
    }

    Ort::Value mel_vocoder_tensor = Ort::Value::CreateTensor<float>(
        g_mem_info, vocoder_mel_data, mel_input_shape[0] * mel_input_shape[1] * mel_input_shape[2],
        mel_input_shape.data(), mel_input_shape.size());

    const char* vocoder_input_names[] = {"mels"};
    const char* vocoder_output_names[] = {"mag", "x", "y"};

    std::vector<Ort::Value> vocoder_inputs;
    vocoder_inputs.push_back(std::move(mel_vocoder_tensor));

    if (cfg.debug) std::cout << "Running vocoder inference..." << std::endl;
    auto t_vocos_start = std::chrono::high_resolution_clock::now();
    auto vocoder_outputs = g_vocoder_session->Run(
        Ort::RunOptions{nullptr},
        vocoder_input_names, vocoder_inputs.data(), vocoder_inputs.size(),
        vocoder_output_names, 3);
    auto t_vocos_end = std::chrono::high_resolution_clock::now();
    result.vocos_ms = std::chrono::duration<double, std::milli>(t_vocos_end - t_vocos_start).count();
    if (cfg.debug) std::cout << "Vocoder inference done in " << (result.vocos_ms / 1000.0) << "s" << std::endl;
    fprintf(stderr, "[TIMING] Vocos          : %.1f ms\n", result.vocos_ms);

    // Reconstruct waveform via ISTFT from (mag, x, y)
    float* mag_data = vocoder_outputs[0].GetTensorMutableData<float>();
    float* x_data = vocoder_outputs[1].GetTensorMutableData<float>();
    float* y_data = vocoder_outputs[2].GetTensorMutableData<float>();
    auto mag_info = vocoder_outputs[0].GetTensorTypeAndShapeInfo();
    auto mag_shape = mag_info.GetShape();
    // mag_shape = [1, n_bins(=n_fft/2+1), frames]
    int n_bins = (int)mag_shape[1];
    int n_fft = 2 * (n_bins - 1);
    int hop = 256;
    int64_t vocos_frames = mag_shape[2];
    if (cfg.debug) {
        std::cout << "Vocos output: [" << mag_shape[0] << ", " << n_bins << ", " << vocos_frames
                  << "] (n_fft=" << n_fft << ", hop=" << hop << ")" << std::endl;
    }

    GRAIN_LOGI("vocoder output: frames=%d n_bins=%d n_fft=%d", (int)vocos_frames, n_bins, n_fft);
    std::vector<float> audio = vocos_istft(mag_data, x_data, y_data,
                                           n_fft, hop, (int)vocos_frames);
    GRAIN_LOGI("istft: samples=%d (vocoder frames=%d hop=%d n_fft=%d)",
               (int)audio.size(), (int)vocos_frames, hop, n_fft);

    // Ali approved the «پ» sample (2026-10-03): the model's first phones carry a sharp attack
    // transient which the ear reports as a «تیک», so every synthesized piece gets a 20 ms onset
    // trim and a 30 ms raised-cosine ramp at both ends. Done HERE so all callers behave alike —
    // including the screen-reader path, where each sentence is its own native call.
    // The config's sample_rate has been seen corrupted at runtime on the device (values like
    // 6029312 / -292686848) while the engine itself synthesizes at 16 kHz. Validate it instead of
    // trusting it: a bad rate only ever produces a bad trim.
    const int eff_rate = (cfg.sample_rate >= 8000 && cfg.sample_rate <= 48000) ? cfg.sample_rate : 16000;
    if (eff_rate != cfg.sample_rate) {
        GRAIN_LOGI("WARNING: cfg.sample_rate=%d is out of range — using %d for trim/fade",
                   cfg.sample_rate, eff_rate);
    }
    trimOnset(audio, eff_rate);
    fadeEdges(audio, eff_rate);
    GRAIN_LOGI("after trim/fade: samples=%d (sr=%d, trim_ms=%.1f, fade_ms=%.1f)",
               (int)audio.size(), cfg.sample_rate,
               envSeconds("GRAIN_TRIM_ONSET_MS", 20.0), envSeconds("GRAIN_FADE_MS", 30.0));
    int64_t num_samples = (int64_t)audio.size();
    if (cfg.debug) std::cout << "Wave shape: [1, " << num_samples << "]" << std::endl;

    // audio_out (JNI path): return samples directly, skip WAV file write.
    if (audio_out) *audio_out = std::move(audio);

    if (!audio_out) {
        std::string actual_output = output_path.empty() ? "output.wav" : output_path;
        write_wav(actual_output, audio, cfg.sample_rate);
        result.output_path = actual_output;
    }
    result.ok = true;
    result.duration_secs = num_samples / (double)cfg.sample_rate;
    result.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_total_start).count();

    if (!quiet) {
        if (cfg.debug) std::cout << "\n=== Done ===" << std::endl;
        if (!audio_out) std::cout << "Output: " << result.output_path << std::endl;
        std::cout << "Duration: " << result.duration_secs << "s" << std::endl;

        fprintf(stderr, "\n[TIMING] ===== Summary =====\n");
        fprintf(stderr, "[TIMING] NormalizeText : %.1f ms\n", result.norm_ms);
        fprintf(stderr, "[TIMING] MatchaTTS      : %.1f ms\n", result.matcha_ms);
        fprintf(stderr, "[TIMING] Vocos          : %.1f ms\n", result.vocos_ms);
        fprintf(stderr, "[TIMING] Total           : %.1f ms\n", result.total_ms);
    }

    return result;
}


// ============================================================================
// Ali 2026-10-03: «متن بین جداکنندهها باید مستقل تبدیل به گفتار بشه» — the text between two
// separators is converted to speech on its OWN: own NormalizeText, own phone run, own model
// run. The pieces are joined with silence whose length follows the separator kind, because the
// model cannot produce silence itself (its alphabet is phones only, no silence symbol).
//
// There is deliberately NO character cap here (Ali: «سقف نباید بگذاری… وگرنه یک بخش از متن
// حذف میشه»): only real separators end a piece, so no part of the text can be dropped. The
// screen-reader path in the app does have a safety chunk size, and there it breaks at the last
// word boundary instead of mid-word.
// ============================================================================
struct SynthSegment {
    std::string text;
    int gap_class;   // 0 = end of text, 1 = comma, 2 = semicolon/colon, 3 = sentence end, 4 = newline
};

static std::vector<SynthSegment> splitAtSeparators(const std::string& text) {
    std::vector<SynthSegment> out;
    std::string cur;
    auto flush = [&](int gap) {
        const char* ws = " \t\r\n";
        size_t a = cur.find_first_not_of(ws);
        size_t b = cur.find_last_not_of(ws);
        if (a != std::string::npos && b != std::string::npos && b >= a) {
            out.push_back({cur.substr(a, b - a + 1), gap});
        } else if (gap && !out.empty()) {
            if (gap > out.back().gap_class) out.back().gap_class = gap;
        }
        cur.clear();
    };
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) len = 1;
        const std::string cp = text.substr(i, len);
        int gap = 0;
        if (cp == "\n" || cp == "\xE2\x80\xA9")         gap = 4;
        else if (cp == "," || cp == "\xD8\x8C")         gap = 1;   // , ،
        else if (cp == ";" || cp == ":" ||
                 cp == "\xD8\x9B")                      gap = 2;   // ؛
        else if (cp == "." || cp == "!" || cp == "?" ||
                 cp == "\xD8\x9F" || cp == "\xDB\x94" ||
                 cp == "\xE2\x80\xA6")                  gap = 3;   // ؟ ۔ …
        if (gap) flush(gap); else cur += cp;
        i += len;
    }
    flush(0);
    return out;
}

static double gapSeconds(int gap_class) {
    double base;
    switch (gap_class) {
        case 1:  base = 0.18; break;   // ،  کوتاه
        case 2:  base = 0.28; break;   // ؛  :
        case 3:  base = 0.45; break;   // . ! ? ؟
        case 4:  base = 0.70; break;   // خط جدید / پاراگراف
        default: return 0.0;
    }
    // tune the pause length without a rebuild (1.0 = the table). Ali chose the defaults.
    const char* env = std::getenv("GRAIN_GAP_SCALE");
    double scale = (env && *env) ? std::atof(env) : 1.0;
    if (scale <= 0.0) scale = 1.0;
    return base * scale;
}


// The click (Ali 2026-10-03: «یک صدای تیک اول مکث‌ها هست، اذیت می‌کنه») is a discontinuity: the
// waveform ends at a non-zero value and the very next sample is digital silence. A 5 ms ramp on
// both edges of every piece removes it (measured: the edge jump fell from 3024 to 22).

static SynthResult synthesize(const SynthConfig& cfg, const std::string& text,
                              const std::string& output_path,
                              std::vector<float>* audio_out = nullptr) {
    std::vector<SynthSegment> segs = splitAtSeparators(text);
    if (segs.size() <= 1) {
        return synthesize_one(cfg, text, output_path, audio_out);
    }

    std::vector<float> all;
    double norm = 0, ac = 0, voc = 0;
    int done = 0, failed = 0;
    for (size_t i = 0; i < segs.size(); ++i) {
        std::vector<float> part;
        SynthResult r = synthesize_one(cfg, segs[i].text, "", &part, /*quiet=*/true);
        if (!r.ok || part.empty()) {
            ++failed;
            std::cerr << "[segment] no audio for: " << segs[i].text << std::endl;
            continue;
        }
        all.insert(all.end(), part.begin(), part.end());
        norm += r.norm_ms; ac += r.matcha_ms; voc += r.vocos_ms;
        ++done;
        // silence between pieces only (never inside one, never after the last)
        if (i + 1 < segs.size()) {
            const double g = gapSeconds(segs[i].gap_class);
            if (g > 0) all.insert(all.end(), (size_t)(g * cfg.sample_rate), 0.0f);
        }
    }

    SynthResult agg;
    if (done == 0) {
        agg.ok = false;
        agg.error = "segmented synthesis produced no audio";
        return agg;
    }
    const size_t total = all.size();
    if (audio_out) {
        *audio_out = std::move(all);
    } else {
        const std::string actual = output_path.empty() ? "output.wav" : output_path;
        write_wav(actual, all, cfg.sample_rate);
        agg.output_path = actual;
    }
    agg.ok = true;
    agg.norm_ms = norm; agg.matcha_ms = ac; agg.vocos_ms = voc;
    agg.duration_secs = total / (double)cfg.sample_rate;
    agg.total_ms = (norm + ac + voc);
    std::cout << "Segments: " << segs.size() << " ok=" << done << " failed=" << failed
              << " -> " << agg.duration_secs << "s" << std::endl;
    return agg;
}

// ============================================================================
// Unix socket helpers
// ============================================================================
static ssize_t write_all(int fd, const void* buf, size_t len) {
    size_t total = 0;
    const char* ptr = static_cast<const char*>(buf);
    while (total < len) {
        ssize_t n = write(fd, ptr + total, len - total);
        if (n <= 0) return n;
        total += n;
    }
    return total;
}

static std::string read_line(int fd, int timeout_secs = 10) {
    struct timeval tv;
    tv.tv_sec = timeout_secs;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string line;
    char c;
    while (true) {
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        if (c == '\n') break;
        line += c;
    }
    return line;
}

static bool write_line(int fd, const std::string& line) {
    std::string msg = line + "\n";
    return write_all(fd, msg.data(), msg.size()) > 0;
}

// ============================================================================
// DAEMON MODE
// ============================================================================
static int run_daemon(const SynthConfig& cfg) {
    signal(SIGPIPE, SIG_IGN);

    std::cout << "[DAEMON] Loading all models..." << std::endl;
    if (!load_all_models(cfg)) {
        std::cerr << "[DAEMON] Failed to load models" << std::endl;
        return 1;
    }

    unlink(SOCKET_PATH);

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("[DAEMON] socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("[DAEMON] bind");
        close(server_fd);
        return 1;
    }

    chmod(SOCKET_PATH, 0666);

    if (listen(server_fd, 5) < 0) {
        perror("[DAEMON] listen");
        close(server_fd);
        unlink(SOCKET_PATH);
        return 1;
    }

    std::cerr << "[DAEMON] Ready on " << SOCKET_PATH << std::endl;

    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            perror("[DAEMON] accept");
            continue;
        }

        std::string request = read_line(client_fd, 30);
        if (request.empty()) {
            close(client_fd);
            continue;
        }

        std::string cmd = json_get_str(request, "command");
        if (cmd == "stop") {
            write_line(client_fd, "{\"status\":\"ok\",\"message\":\"shutting down\"}");
            close(client_fd);
            break;
        }

        std::string text = json_get_str(request, "text");
        if (text.empty()) {
            write_line(client_fd, "{\"status\":\"error\",\"message\":\"--text is required\"}");
            close(client_fd);
            continue;
        }

        SynthConfig req_cfg = cfg;
        std::string temp_str = json_get_str(request, "temperature");
        if (!temp_str.empty()) req_cfg.temperature = std::stof(temp_str);
        std::string speed_str = json_get_str(request, "speed");
        if (!speed_str.empty()) req_cfg.speed = std::stof(speed_str);
        std::string sr_str = json_get_str(request, "sample_rate");
        if (!sr_str.empty()) req_cfg.sample_rate = std::stoi(sr_str);
        std::string lang_str = json_get_str(request, "main_lang");
        if (!lang_str.empty()) req_cfg.main_lang_str = lang_str;

        std::string output = json_get_str(request, "output");
        if (output.empty()) output = "output.wav";

        SynthResult result = synthesize(req_cfg, text, output);

        char buf[4096];
        if (result.ok) {
            snprintf(buf, sizeof(buf),
                "{\"status\":\"ok\",\"output\":\"%s\",\"duration\":%.2f,\"norm_ms\":%.1f,\"matcha_ms\":%.1f,\"vocos_ms\":%.1f,\"total_ms\":%.1f}",
                json_esc(result.output_path).c_str(),
                result.duration_secs,
                result.norm_ms, result.matcha_ms, result.vocos_ms, result.total_ms);
        } else {
            snprintf(buf, sizeof(buf),
                "{\"status\":\"error\",\"message\":\"%s\"}",
                json_esc(result.error).c_str());
        }
        write_line(client_fd, std::string(buf));

        close(client_fd);
    }

    close(server_fd);
    unlink(SOCKET_PATH);
    std::cout << "[DAEMON] Stopped" << std::endl;
    return 0;
}

// ============================================================================
// STOP DAEMON
// ============================================================================
static int stop_daemon() {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Daemon is not running (cannot connect to " << SOCKET_PATH << ")" << std::endl;
        close(fd);
        return 1;
    }

    write_line(fd, "{\"command\":\"stop\"}");
    std::string response = read_line(fd, 5);
    std::cout << "Daemon: " << response << std::endl;
    close(fd);
    return 0;
}

// ============================================================================
// CLIENT MODE
// ============================================================================
static bool daemon_is_running() {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    bool running = (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    close(fd);
    return running;
}

static int run_client(const SynthConfig& cfg, const std::string& text,
                       const std::string& output_path, bool play_audio) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect"); close(fd); return 1;
    }
    std::string request = "{\"text\":" + json_str(text);
    if (!output_path.empty()) request += ",\"output\":" + json_str(output_path);
    request += ",\"speed\":" + std::to_string(cfg.speed);
    request += ",\"main_lang\":" + json_str(cfg.main_lang_str);      // the daemon serves FA and EN
    request += ",\"sample_rate\":" + std::to_string(cfg.sample_rate);
    request += "}";
    if (!write_line(fd, request)) { std::cerr << "Error: failed to send\n"; close(fd); return 1; }
    std::string response = read_line(fd, 30);
    close(fd);
    if (response.empty()) { std::cerr << "Error: no response\n"; return 1; }
    std::string st = json_get_str(response, "status");
    if (st == "ok") {
        std::string out = json_get_str(response, "output");
        std::cout << "Output: " << out << "\nDuration: " << json_get_str(response, "duration") << "s\n";
        if (play_audio && !out.empty()) {
            std::string cmd = "ffplay -nodisp -autoexit \"" + out + "\" 2>/dev/null";
            std::system(cmd.c_str());
        }
        return 0;
    }
    std::cerr << "Error from daemon: " << json_get_str(response, "message") << "\n";
    return 1;
}

// ============================================================================
// USAGE
// ============================================================================
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "\n"
              << "  MODES:\n"
              << "    --daemon                    Start background daemon (loads models once, stays resident)\n"
              << "    --stop                      Stop the running daemon\n"
              << "\n"
              << "  SYNTHESIS OPTIONS:\n"
              << "    --text <text>               Text to synthesize (required unless --daemon/--stop)\n"
              << "    --model <path>              GrainSpeech acoustic model ONNX (phones -> mel)\n"
              << "    --vocoder-model <path>      Vocos ONNX (mels -> mag/x/y)\n"
              << "    --symbols <path>            Symbol table (one token per line, line no. = id)\n"
              << "    --espeak-data <path>        espeak-ng-data directory\n"
              << "    --ezafe-onnx <path>         Ezafe model ONNX path\n"
              << "    --ezafe-spiece <path>       Ezafe sentencepiece model path\n"
              << "    --hazm-words <path>         HAZM words.dat path\n"
              << "    --hazm-verbs <path>         HAZM verbs.dat path\n"
              << "    --hazm-stopwords <path>     HAZM stopwords.dat path\n"
              << "    --homograph <path>          Homograph data JSON path\n"
              << "    --shakkelha <path>          Shakkelha ONNX model path\n"
              << "    --mel-transform <path>      80x80 float matrix file (maps matcha log-mel\n"
              << "                                mags to the vocoder's training mel style;\n"
              << "                                e.g. slaney->htk for custom 16k vocoders)\n"
              << "    --output <path>             Output WAV file (default: output.wav)\n"
              << "    --play                      Play audio after generation\n"
              << "    --temperature <float>       Temperature (default: 0.667)\n"
              << "    --speed <float>             Speed: 1.0=1x, 1.5=1.5x, 2.0=2x (default: 1.5)\n"
              << "    --sample-rate <int>         Output sample rate (default: 16000)\n"
              << "    --main-lang <EN|FA|AR>      Main language (default: FA)\n"
              << "                                (CPU only by design)\n"
              << "    --debug                     Print IPA phonemes, normalized text, and pipeline details\n"
              << "    --help                      Show this help\n"
              << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    std::locale::global(std::locale("en_US.UTF-8"));

    // ----- Paths (passed externally via CLI — no hardcoded defaults) -----
    SynthConfig cfg;
    // Model paths: REQUIRED via CLI flags (--matcha-model, --vocoder-model,
    // --tokens, --espeak-data). The caller (tts.py) passes them explicitly.
    // NormalizeText assets live relative to the binary (./assets/) — these
    // are part of the match_tts_infer repo, not external models.
    cfg.ezafe_onnx     = "./assets/ezafe_model.onnx";
    cfg.ezafe_spiece   = "./assets/ezafe_spiece.model";
    cfg.hazm_words     = "./assets/hazm_words.dat";
    cfg.hazm_verbs     = "./assets/hazm_verbs.dat";
    cfg.hazm_stopwords = "./assets/hazm_stopwords.dat";
    cfg.homograph_data = "./assets/homograph_data.json";
    cfg.shakkelha_onnx = "./assets/shakkelha.onnx";

    std::string text;
    std::string output_wav;
    bool play_audio = false;
    bool daemon_mode = false;
    bool stop_mode = false;

    // ----- Parse CLI args -----
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto require_val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " requires a value" << std::endl;
                exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--text")               text = require_val("--text");
        else if (arg == "--model" || arg == "--acoustic-model")
                                           cfg.acoustic_model = require_val(arg.c_str());
        else if (arg == "--vocoder-model") cfg.vocoder_model = require_val("--vocoder-model");
        else if (arg == "--symbols")       { cfg.symbols_file = require_val("--symbols");
                                             cfg.tokens_file = cfg.symbols_file; }
        else if (arg == "--espeak-data")   cfg.espeak_data = require_val("--espeak-data");
        else if (arg == "--ezafe-onnx")    cfg.ezafe_onnx = require_val("--ezafe-onnx");
        else if (arg == "--ezafe-spiece")  cfg.ezafe_spiece = require_val("--ezafe-spiece");
        else if (arg == "--hazm-words")    cfg.hazm_words = require_val("--hazm-words");
        else if (arg == "--hazm-verbs")    cfg.hazm_verbs = require_val("--hazm-verbs");
        else if (arg == "--hazm-stopwords") cfg.hazm_stopwords = require_val("--hazm-stopwords");
        else if (arg == "--homograph")     cfg.homograph_data = require_val("--homograph");
        else if (arg == "--shakkelha")     cfg.shakkelha_onnx = require_val("--shakkelha");
        else if (arg == "--mel-transform") cfg.mel_transform = require_val("--mel-transform");
        else if (arg == "--output")        output_wav = require_val("--output");
        else if (arg == "--play")          play_audio = true;
        else if (arg == "--temperature")   cfg.temperature = std::stof(require_val("--temperature"));
        else if (arg == "--speed")         cfg.speed = std::stof(require_val("--speed"));
        else if (arg == "--sample-rate")   cfg.sample_rate = std::stoi(require_val("--sample-rate"));
        else if (arg == "--main-lang")     cfg.main_lang_str = require_val("--main-lang");
        else if (arg == "--gpu")           cfg.use_gpu = true;
        else if (arg == "--debug")         cfg.debug = true;
        else if (arg == "--daemon")        daemon_mode = true;
        else if (arg == "--stop")          stop_mode = true;
        else if (arg == "--help")          { print_usage(argv[0]); return 0; }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // ----- Dispatch -----
    if (daemon_mode) {
        return run_daemon(cfg);
    }

    if (stop_mode) {
        return stop_daemon();
    }

    // Default --text: if a daemon is already running, reuse it (fast path);
    // otherwise synthesize directly in this process (one-shot — no daemon).
    // Use --daemon to start the resident server explicitly; --stop to end it.
    if (text.empty()) {
        std::cerr << "Error: --text is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if (daemon_is_running()) {
        return run_client(cfg, text, output_wav, play_audio);
    }

    if (!load_all_models(cfg)) {
        std::cerr << "Error: failed to load models" << std::endl;
        return 1;
    }

    SynthResult result = synthesize(cfg, text, output_wav);
    if (!result.ok) {
        std::cerr << "Error: " << result.error << std::endl;
        return 1;
    }

    if (play_audio && !result.output_path.empty()) {
        std::string cmd = "ffplay -nodisp -autoexit \"" + result.output_path + "\" 2>/dev/null";
        std::system(cmd.c_str());
    }
    return 0;
}
