
# ✅ Clean build with Release config

dotnet clean
rm -rf bin/ obj/
dotnet restore
dotnet build --configuration Release --no-incremental

# ✅ Set CPU governor (if on Linux)

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# ✅ Optional: disable hyperthreading for dedicated inference

# for cpu in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do

# echo 0 | sudo tee ${cpu/thread_siblings_list/online} 2>/dev/null

# done

# ✅ Run

# 1. Bebaskan RAM/swap sebelum test

sudo sync && sudo sysctl vm.drop_caches=3
sudo swapoff -a && sudo swapon -a
free -h  # pastikan swap turun

# 2. Restart dengan thread fix

dotnet run --configuration Release

---

## Testing in bash Terminal

# ── Warmup dulu (WAJIB — biar JIT/TieredPGO compile hot paths) ─────────────

echo "=== WARMUP ===" && \
curl -s -N -X POST <http://localhost:5034/v1/chat/completions> \
  -H "Authorization: Bearer API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi bro"}],"max_tokens":30,"stream":false }' | tail -3

echo ""
echo "=== BENCHMARK ==="

# ── Benchmark utama — SSE streaming, hitung waktu real ──────────────────────

START=$(date +%s%3N)
curl -X POST http://localhost:5034/v1/chat/completions \
  -H "Authorization: Bearer API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello bro, kamu bisa bahasa indonesia tidak? kalau bisa, coba jelaskan Vect<> dalam rust programming."}],"stream":true}'
    | tee /tmp/bench_out.txt
  END=$(date +%s%3N)

echo ""
TOKENS=$(grep -o '"tokens_used":[0-9]*' /tmp/bench_out.txt | tail -1 | grep -o '[0-9]*')
ELAPSED_MS=$((END - START))
echo "────────────────────────────────────"
echo "Tokens generated : $TOKENS"
echo "Total time       : ${ELAPSED_MS}ms"
if [ -n "$TOKENS" ] && [ "$TOKENS" -gt 0 ]; then
  echo "Speed            : $(echo "scale=2; $TOKENS * 1000 / $ELAPSED_MS" | bc) tok/s"
fi

---

# Streaming (default)

curl -X POST <http://localhost:5005/v1/chat/completions> \
  -H "Authorization: Bearer API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"halo bro, dirimu bisa bahasa Indonesia tidak bro.?"}],"stream":true}'

# Non-streaming (dapat full JSON sekaligus)

curl -X POST <http://localhost:5005/v1/chat/completions> \
  -H "Authorization: Bearer API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"halo bro"}],"stream":false}'

# List models (untuk OpenWebUI/LangChain auto-detect)

curl <http://localhost:5005/v1/models> \
  -H "Authorization: Bearer API KEY"

curl <http://localhost:5005/health>
----
