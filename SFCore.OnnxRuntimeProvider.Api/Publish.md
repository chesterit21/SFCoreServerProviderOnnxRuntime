### Publish

dotnet publish -r win-x64 -c Release --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true

dotnet publish -r linux-x64 -c Release --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true

# Kasih permission execute dulu

chmod +x ./publish/SFCore

# Jalankan langsung

./publish/SFCore

# Atau kalau pakai systemd (equivalent NSSM di Linux)

sudo nano /etc/systemd/system/sfcore.service

[Unit]
Description=SFCore ONNX Inference Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/publish
ExecStart=/path/to/publish/SFCore
Restart=always
RestartSec=5
Environment=ASPNETCORE_ENVIRONMENT=Production
Environment=ASPNETCORE_URLS=<http://0.0.0.0:5005>

[Install]
WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl enable sfcore
sudo systemctl start sfcore
sudo systemctl status sfcore

```

---

**Satu catatan penting untuk Linux:**

Di `HardwareDetector.cs` code kita, kalau VM detection via WMI gagal di Linux (karena WMI itu Windows-only), dia akan fallback gracefully — tapi pastikan ada ini di log startup:
```

[Hardware] VM detected via WMI manufacturer: ...
