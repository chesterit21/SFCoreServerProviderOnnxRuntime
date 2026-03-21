# SFCore.OnnxRuntimeProvider.Api
[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/chesterit21/SFCoreServerProviderOnnxRuntime)
[**Bahasa Indonesia**](#bahasa-indonesia) | [**English**](#english)

---

<div id="bahasa-indonesia"></div>

## 🇮🇩 Bahasa Indonesia

API Provider berbasis ASP.NET Core yang dirancang khusus untuk menjalankan model AI dalam format **ONNX** menggunakan **ONNX Runtime** dengan performa tinggi dan optimasi perangkat keras yang handal.

### 🚀 Fitur Utama

- **Performa Tinggi**: Inferensi model machine learning yang dioptimalkan langsung oleh ONNX Runtime.
- **REST API Siap Pakai**: Integrasi mudah dengan aplikasi eksternal melalui endpoint HTTP/JSON.
- **Arsitektur Fleksibel**: Dirancang untuk menangani berbagai seri model ONNX (khususnya seri Qwen).
- **Deteksi Perangkat Keras Otomatis**: Optimasi cerdas berdasarkan spesifikasi hardware pengguna.

### 🛠️ Prasyarat

- **[.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)** (atau versi terbaru yang kompatibel).
- **Python** (Opsional): Untuk menjalankan skrip pengujian tambahan seperti `debug_request.py`.

### 📦 Pengaturan Model AI ONNX

**PENTING**: Sangat disarankan mengunduh model dari **ONNX Community** karena konfigurasi dan Tokenizer telah disesuaikan di dalam kode.

**Rekomendasi Model**: [onnx-community/Qwen3.5-4B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX/tree/main/onnx) (Versi: `q4f16`, `q4`, atau `int8`).

#### Daftar File Wajib:
- `decoder_model_merged_q4f16.onnx` (& `.onnx_data` jika > 2GB)
- `encoder_model_merged_q4f16.onnx` (& `.onnx_data` jika > 2GB)
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`
- *Opsional*: `vision_encoder_q4f16.onnx` (untuk dukungan Vision/Gambar).

> [!IMPORTANT]
> Jangan lupa untuk memperbarui path root folder model di file `appsettings.json`.

### 💡 Optimasi & Kustomisasi

Jika Anda ingin performa yang lebih optimal:
1. Bagikan kode `HardwareDetector.cs`, `Program.cs`, dan `Qwen35inferenceengine.cs` ke AI (Claude/Gemini/ZAI).
2. Sertakan **informasi spesifikasi perangkat keras (GPU/CPU/RAM)** Anda.
3. Minta AI untuk mengoptimalkan parameter inisialisasi agar sesuai dengan hardware Anda demi efisiensi maksimal.

### 📊 Hasil Benchmark & Performa

Proyek ini telah diuji secara ekstensif pada hardware kelas server (Xeon v4) dan berhasil menangani **64,000 tokens context window** dengan stabil.

- **Konteks Panjang**: Mendukung hingga 64K tokens dengan degradasi performa minimal.
- **Efisiensi**: Arsitektur hybrid memberikan kecepatan prefill yang konsisten bahkan pada context besar.
- **Melampaui Standar**: Berbeda dengan **llama.cpp** atau **Ollama** yang seringkali tidak stabil atau sangat lambat pada CPU untuk konteks di atas 8K, proyek ini berhasil menjalankan **64K context secara penuh** dengan stabil pada hardware tahun 2016.

> [!TIP]
> Lihat laporan performa lengkap dan statistik benchmark di: **[BENCHMARK.md](BENCHMARK.md)**

---

<div id="english"></div>

## 🇺🇸 English

An ASP.NET Core-based API Provider specifically designed to run AI models in **ONNX** format using **ONNX Runtime**, focusing on high performance and reliable hardware optimization.

### 🚀 Key Features

- **High Performance**: Machine learning model inference optimized directly by ONNX Runtime.
- **Ready-to-use REST API**: Seamless integration with external applications via HTTP/JSON endpoints.
- **Extensible Architecture**: Designed to handle various ONNX model series (specifically tailored for Qwen).
- **Auto Hardware Detection**: Intelligent optimization based on the user's hardware specifications.

### 🛠️ Prerequisites

- **[.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)** (or relevant latest versions).
- **Python** (Optional): For running additional testing scripts like `debug_request.py`.

### 📦 ONNX AI Model Setup

**IMPORTANT**: It is highly recommended to download models from the **ONNX Community** repository as configurations and Tokenizers are already synchronized with the codebase.

**Recommended Model**: [onnx-community/Qwen3.5-4B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX/tree/main/onnx) (Versions: `q4f16`, `q4`, or `int8`).

#### Required Files:
- `decoder_model_merged_q4f16.onnx` (& `.onnx_data` if > 2GB)
- `encoder_model_merged_q4f16.onnx` (& `.onnx_data` if > 2GB)
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`
- *Optional*: `vision_encoder_q4f16.onnx` (for Vision/Image support).

> [!IMPORTANT]
> Remember to update the model folder root path in `appsettings.json`.

### 💡 Optimization & Customization

To achieve maximum performance:
1. Share the `HardwareDetector.cs`, `Program.cs`, and `Qwen35inferenceengine.cs` files with an AI (Claude/Gemini/ZAI).
2. Provide your **hardware specifications (GPU/CPU/RAM)**.
3. Ask the AI to optimize the initialization parameters to match your specific hardware for peak efficiency.

### 📊 Benchmarks & Performance

This project has been extensively tested on server-grade hardware (Xeon v4) and successfully handles a **64,000 tokens context window** with high stability.

- **Long Context**: Supports up to 64K tokens with minimal performance degradation.
- **Efficiency**: The hybrid architecture ensures consistent prefill speeds even at large context scales.
- **Beyond Industry Standards**: Unlike **llama.cpp** or **Ollama**, which often encounter instability or severe slowdowns on CPU for contexts exceeding 8K, this project successfully processes a **full 64K context** with stability on hardware from 2016.

> [!TIP]
> Read the full performance report and benchmark statistics at: **[BENCHMARK.md](BENCHMARK.md)**

---

## ⚙️ Quick Start / Mulai Cepat

1. **Clone Repository**:
   ```bash
   git clone https://github.com/USERNAME/SFCoreServerProviderOnnxRuntime.git
   cd SFCoreServerProviderOnnxRuntime
   ```

2. **Restore Dependencies**:
   ```bash
   dotnet restore SFCore.OnnxRuntimeProvider.Api
   ```

3. **Run Application**:
   ```bash
   dotnet run --project SFCore.OnnxRuntimeProvider.Api
   ```

> [!NOTE]
> Default URL: `http://localhost:5034` (or as configured in `appsettings.json`).

## 📖 API Documentation (Swagger)

Once running, access the **Swagger UI** at:
`http://localhost:<PORT>/swagger`

## 🤝 Contributing / Berkontribusi

We welcome contributions! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License / Lisensi

Distributed under the [MIT License](LICENSE).
