#include "normalize.h"
#include "grain_phones.h"
#include "language_detector/language_detector.h"

#include <onnxruntime_cxx_api.h>

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
    std::string lexicon_dir;      // infer_lex_fa.txt / infer_lex_en.txt for the phone conversion
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
static std::map<std::string, std::string> g_lexicon;   // word -> model phone tokens
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
    g_session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Keep resident memory bounded: without these, ORT's CPU arena holds all
    // intermediate activations and never returns memory to the OS (~4 GB RSS).
    g_session_opts.DisableCpuMemArena();
    g_session_opts.DisableMemPattern();

    if (read_token_map(cfg.symbols_file, g_token_to_id, g_id_to_token) != 0) {
        return false;
    }

    g_lang_tag = (cfg.main_lang_str == "EN" || cfg.main_lang_str == "en") ? "en" : "fa";
    if (!cfg.lexicon_dir.empty()) {
        g_lexicon = grain::loadInferenceLexicon(cfg.lexicon_dir, g_lang_tag);
        std::cout << "Loaded " << g_lexicon.size() << " lexicon entries ("
                  << g_lang_tag << ") from " << cfg.lexicon_dir << std::endl;
    }

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
// Core synthesis
// ============================================================================
static SynthResult synthesize(const SynthConfig& cfg, const std::string& text,
                               const std::string& output_path,
                               std::vector<float>* audio_out = nullptr) {
    SynthResult result;
    auto t_total_start = std::chrono::high_resolution_clock::now();

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

    std::string phone_str = grain::toModelPhones(normalized_text, ipa_text, g_lexicon, g_lang_tag);
    std::vector<std::string> phones;
    {
        std::istringstream ss(phone_str);
        std::string tk;
        while (ss >> tk) phones.push_back(tk);
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

    std::vector<float> audio = vocos_istft(mag_data, x_data, y_data,
                                           n_fft, hop, (int)vocos_frames);
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

    if (cfg.debug) std::cout << "\n=== Done ===" << std::endl;
    if (!audio_out) std::cout << "Output: " << result.output_path << std::endl;
    std::cout << "Duration: " << result.duration_secs << "s" << std::endl;

    fprintf(stderr, "\n[TIMING] ===== Summary =====\n");
    fprintf(stderr, "[TIMING] NormalizeText : %.1f ms\n", result.norm_ms);
    fprintf(stderr, "[TIMING] MatchaTTS      : %.1f ms\n", result.matcha_ms);
    fprintf(stderr, "[TIMING] Vocos          : %.1f ms\n", result.vocos_ms);
    fprintf(stderr, "[TIMING] Total           : %.1f ms\n", result.total_ms);

    return result;
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
              << "    --lexicon-dir <dir>         infer_lex_fa.txt / infer_lex_en.txt for the phone conversion\n"
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
        else if (arg == "--lexicon-dir")   cfg.lexicon_dir = require_val("--lexicon-dir");
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
