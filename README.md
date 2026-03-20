# SFCore.OnnxRuntimeProvider.Api

API Provider berbasis ASP.NET Core untuk menjalankan model AI menggunakan ONNX Runtime secara cepat dan handal. 

## 🚀 Fitur Utama
- **High Performance**: Menjalankan inferensi model machine learning dengan optimasi dari ONNX Runtime.
- **REST API**: Mudah diintegrasikan dengan aplikasi lain via endpoint HTTP/JSON.
- **Extensible Architecture**: Ditujukan untuk dapat menangani berbagai jenis model ONNX.

## 🛠️ Prasyarat
- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) (atau versi yang relevan)
- _Opsional_: Python untuk menjalankan skrip pengujian (seperti `debug_request.py`)

## ⚙️ Instalasi dan Menjalankan Proyek

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/USERNAME/SFCoreServerProviderOnnxRuntime.git
   cd SFCoreServerProviderOnnxRuntime
   ```

2. **Restore dependensi NuGet:**
   ```bash
   dotnet restore SFCore.OnnxRuntimeProvider.Api
   ```

3. **Jalankan Aplikasi:**
   ```bash
   dotnet run --project SFCore.OnnxRuntimeProvider.Api
   ```

Aplikasi secara default dapat diakses melalui URL (seperti `http://localhost:5000` atau sesuai dengan konfigurasi yang ada di `appsettings.json`).

## 📖 Dokumentasi & Testing (Swagger)
Setelah aplikasi berjalan, Anda bisa mengakses antarmuka **Swagger UI** melalui browser di:
`http://localhost:<PORT>/swagger` 
Ini berguna untuk melihat spesifikasi endpoint, request payload, dan langsung melakukan pengujian API (Test endpoint).

## 🤝 Berkontribusi
Kami menyambut kontribusi dari komunitas! Jika Anda tertarik, silakan pelajari panduan kontribusi di file `CONTRIBUTING.md`.

## 📄 Lisensi
Proyek ini berlisensi di bawah [MIT License](LICENSE).
