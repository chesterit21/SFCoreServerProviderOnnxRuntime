# SFCore.OnnxRuntimeProvider.Api

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
