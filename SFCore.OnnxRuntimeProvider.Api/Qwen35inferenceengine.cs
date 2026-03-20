// ============================================================
//  Qwen35InferenceEngine.cs — Qwen3.5 Hybrid Attention Inference
//
//  Optimized for: VMware shared-host Windows VPS, 16 vCPU, 64 GB RAM
//
//  VM-specific optimizations vs bare metal version:
//  1. PrefillChunkSize reduced 512→256:
//     On a shared host, a single ORT call blocking 12 threads for 1-2s
//     causes ASP.NET thread pool starvation. 256 tokens ≈ 0.5-0.8s,
//     keeping the thread pool healthy for concurrent requests.
//  2. ArrayPool sizing: slightly more conservative to avoid LOH pressure
//     on Windows VM where GC pauses are more impactful.
//  3. SessionOptions tuned for Windows VM (see BuildSessionOptions).
//
//  Confirmed tensor dtypes for q4f16 model:
//    inputs_embeds          → float32  (from embed_tokens output)
//    attention_mask         → int64
//    position_ids           → int64    [3, 1, seq] mrope
//    past_conv.N            → float16  (linear attn state, fixed size)
//    past_recurrent.N       → float16  (linear attn state, fixed size)
//    past_key_values.N.key  → float16  (full attn KV cache, grows)
//    past_key_values.N.val  → float16  (full attn KV cache, grows)
//    logits output          → float32
//    present_conv.N         → float16
//    present_recurrent.N    → float16
//    present.N.key/value    → float16
// ============================================================

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;
using System.Runtime.CompilerServices;

public sealed class Qwen35InferenceEngine : IDisposable
{
    private readonly ModelConfig _cfg;
    private readonly ILogger _logger;
    private readonly InferenceSession _embedSession;
    private readonly InferenceSession _decoderSession;
    private readonly string[] _outputNames;
    private readonly float[] _logitsScratch;

    private int _convStateSize = 6144;
    private int _convKernelDim = 4;
    private int _recurrentNH = 16;
    private int _recurrentKD = 128;

    // ── VM-tuned prefill chunk size ───────────────────────────────────────────
    // 256 instead of 512: each ORT call ~0.5-0.8s on 12 threads.
    // Keeps ASP.NET thread pool responsive on shared-host VM.
    // Also yields more frequently to the Task scheduler, allowing
    // cancellation tokens to be checked more often (better request abort).
    private const int PrefillChunkSize = 256;

    public Qwen35InferenceEngine(ModelConfig cfg, ILogger logger)
    {
        _cfg = cfg;
        _logger = logger;

        var opts = BuildSessionOptions(cfg);

        _logger.LogInformation("[Qwen35] Loading embed_tokens: {F}",
            Path.GetFileName(cfg.EmbedTokensFile!));
        _embedSession = new InferenceSession(cfg.EmbedTokensFile!, opts);

        _logger.LogInformation("[Qwen35] Loading decoder: {F}",
            Path.GetFileName(cfg.DecoderModelFile));
        _decoderSession = new InferenceSession(cfg.DecoderModelFile, opts);

        _outputNames = BuildOutputNames(cfg);
        _logitsScratch = GC.AllocateArray<float>(cfg.VocabSize, pinned: true);

        InspectSessionInputs();

        _logger.LogInformation("[Qwen35] Ready — Vocab={V} FullAttnLayers=[{FA}]",
            cfg.VocabSize, string.Join(",", cfg.FullAttentionLayers));
    }

    // =========================================================================
    //  PUBLIC API
    // =========================================================================

    public async IAsyncEnumerable<int> GenerateAsync(
        int[] inputIds,
        int maxNewTokens = 512,
        float temperature = 0.7f,
        float topP = 0.9f,
        int topK = 20,
        IEnumerable<int>? stopTokenIds = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var stopIds = stopTokenIds as HashSet<int>
                      ?? (stopTokenIds?.ToHashSet() ?? _cfg.EosTokenIds);

        var state = new HybridState(_cfg, _convStateSize, _convKernelDim,
                                    _recurrentNH, _recurrentKD);

        // ── Inference profiling ───────────────────────────────────────────────
        // Tracks prefill vs decode time separately.
        // Prefill = processing the entire input prompt (slow, scales with ctx).
        // Decode  = generating each new output token (fast, ~constant per token).
        // Having both numbers in the log is essential for diagnosing performance:
        //   "prompt=8000 completion=50" but took 4 minutes → all time in prefill
        //   "prompt=100 completion=500" but took 4 minutes → all time in decode
        var swTotal = System.Diagnostics.Stopwatch.StartNew();
        var swPrefill = System.Diagnostics.Stopwatch.StartNew();
        int promptLen = inputIds.Length;

        try
        {
            int pastLen = 0;

            int nextToken;
            if (inputIds.Length <= PrefillChunkSize)
            {
                nextToken = RunStep(inputIds, state, ref pastLen, temperature, topP, topK);
            }
            else
            {
                // Chunked prefill with Task.Yield between chunks.
                // On a shared VMware host, yielding here allows:
                //   - Other ASP.NET requests to be processed
                //   - CancellationToken to be observed promptly
                //   - VMware scheduler to balance vCPU load across host
                int pos = 0;
                while (pos + PrefillChunkSize < inputIds.Length)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    RunPrefillChunk(inputIds[pos..(pos + PrefillChunkSize)], state, ref pastLen);
                    pos += PrefillChunkSize;
                    await Task.Yield();
                }
                nextToken = RunStep(inputIds[pos..], state, ref pastLen, temperature, topP, topK);
            }

            // ── Log prefill stats ─────────────────────────────────────────────
            swPrefill.Stop();
            double prefillSec = swPrefill.Elapsed.TotalSeconds;
            _logger.LogInformation(
                "[Qwen35] Prefill — {P} tokens in {S:F1}s ({TPS:F1} tok/s)",
                promptLen, prefillSec, promptLen / Math.Max(prefillSec, 0.001));

            // ── Decode phase ──────────────────────────────────────────────────
            var swDecode = System.Diagnostics.Stopwatch.StartNew();
            int decodeCount = 0;

            if (stopIds.Contains(nextToken)) yield break;
            yield return nextToken;
            decodeCount++;

            for (int step = 0; step < maxNewTokens - 1; step++)
            {
                if (cancellationToken.IsCancellationRequested) yield break;
                nextToken = RunStep([nextToken], state, ref pastLen, temperature, topP, topK);
                if (stopIds.Contains(nextToken)) yield break;
                yield return nextToken;
                decodeCount++;
                await Task.Yield();
            }

            // ── Log decode stats ──────────────────────────────────────────────
            swDecode.Stop();
            swTotal.Stop();
            double decodeSec = swDecode.Elapsed.TotalSeconds;
            _logger.LogInformation(
                "[Qwen35] Decode — {D} tokens in {S:F1}s ({TPS:F1} tok/s) | total={T:F1}s",
                decodeCount, decodeSec,
                decodeCount / Math.Max(decodeSec, 0.001),
                swTotal.Elapsed.TotalSeconds);
        }
        finally { state.Dispose(); }
    }

    // =========================================================================
    //  PREFILL CHUNK — process tokens without sampling output
    // =========================================================================

    private void RunPrefillChunk(int[] tokenIds, HybridState state, ref int pastLen)
    {
        int seqLen = tokenIds.Length;
        int totalSeq = pastLen + seqLen;

        long[] inputIdsArr = ArrayPool<long>.Shared.Rent(seqLen);
        float[] embeds;
        try
        {
            for (int i = 0; i < seqLen; i++) inputIdsArr[i] = tokenIds[i];
            var embedInputs = new List<NamedOnnxValue>(1)
            {
                NamedOnnxValue.CreateFromTensor("input_ids",
                    new DenseTensor<long>(inputIdsArr.AsMemory(0, seqLen), [1, seqLen]))
            };
            using var embedResult = _embedSession.Run(embedInputs);
            int embedSize = seqLen * _cfg.HiddenSize;
            embeds = ArrayPool<float>.Shared.Rent(embedSize);
            var et = embedResult[0].AsTensor<float>();
            if (et is DenseTensor<float> det)
                det.Buffer.Span.Slice(0, embedSize).CopyTo(embeds.AsSpan(0, embedSize));
            else
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < _cfg.HiddenSize; h++)
                        embeds[s * _cfg.HiddenSize + h] = et[0, s, h];
        }
        finally { ArrayPool<long>.Shared.Return(inputIdsArr); }

        long[] maskArr = ArrayPool<long>.Shared.Rent(totalSeq);
        long[] posArr = ArrayPool<long>.Shared.Rent(3 * seqLen);
        try
        {
            for (int i = 0; i < totalSeq; i++) maskArr[i] = 1L;
            for (int plane = 0; plane < 3; plane++)
                for (int i = 0; i < seqLen; i++)
                    posArr[plane * seqLen + i] = pastLen + i;

            var inputs = BuildDecoderInputs(embeds, seqLen, maskArr, posArr,
                                            totalSeq, state, pastLen);
            using var results = _decoderSession.Run(inputs, _outputNames);

            UpdateState(results, state, totalSeq);
            pastLen = totalSeq;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(embeds);
            ArrayPool<long>.Shared.Return(maskArr);
            ArrayPool<long>.Shared.Return(posArr);
        }
    }

    // =========================================================================
    //  CORE STEP
    // =========================================================================

    private int RunStep(
        int[] tokenIds,
        HybridState state,
        ref int pastLen,
        float temperature,
        float topP,
        int topK)
    {
        int seqLen = tokenIds.Length;
        int totalSeq = pastLen + seqLen;

        long[] inputIdsArr = ArrayPool<long>.Shared.Rent(seqLen);
        float[] embeds;
        try
        {
            for (int i = 0; i < seqLen; i++) inputIdsArr[i] = tokenIds[i];

            var embedInputs = new List<NamedOnnxValue>(1)
            {
                NamedOnnxValue.CreateFromTensor("input_ids",
                    new DenseTensor<long>(inputIdsArr.AsMemory(0, seqLen), [1, seqLen]))
            };

            using var embedResult = _embedSession.Run(embedInputs);
            int embedSize = seqLen * _cfg.HiddenSize;
            embeds = ArrayPool<float>.Shared.Rent(embedSize);

            var et = embedResult[0].AsTensor<float>();
            if (et is DenseTensor<float> det)
                det.Buffer.Span.Slice(0, embedSize).CopyTo(embeds.AsSpan(0, embedSize));
            else
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < _cfg.HiddenSize; h++)
                        embeds[s * _cfg.HiddenSize + h] = et[0, s, h];
        }
        finally { ArrayPool<long>.Shared.Return(inputIdsArr); }

        long[] maskArr = ArrayPool<long>.Shared.Rent(totalSeq);
        long[] posArr = ArrayPool<long>.Shared.Rent(3 * seqLen);

        try
        {
            for (int i = 0; i < totalSeq; i++) maskArr[i] = 1L;
            for (int plane = 0; plane < 3; plane++)
                for (int i = 0; i < seqLen; i++)
                    posArr[plane * seqLen + i] = pastLen + i;

            var inputs = BuildDecoderInputs(embeds, seqLen, maskArr, posArr,
                                            totalSeq, state, pastLen);

            using var results = _decoderSession.Run(inputs, _outputNames);

            var logitsF16 = results[0].AsTensor<Float16>();
            int offset = (seqLen - 1) * _cfg.VocabSize;
            if (logitsF16 is DenseTensor<Float16> dl)
            {
                var src = dl.Buffer.Span.Slice(offset, _cfg.VocabSize);
                for (int i = 0; i < _cfg.VocabSize; i++)
                    _logitsScratch[i] = (float)src[i];
            }
            else
            {
                for (int i = 0; i < _cfg.VocabSize; i++)
                    _logitsScratch[i] = (float)logitsF16[0, seqLen - 1, i];
            }

            UpdateState(results, state, totalSeq);
            pastLen = totalSeq;
            return Sampler.Sample(_logitsScratch, temperature, topP, topK);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(embeds);
            ArrayPool<long>.Shared.Return(maskArr);
            ArrayPool<long>.Shared.Return(posArr);
        }
    }

    // =========================================================================
    //  BUILD DECODER INPUTS
    // =========================================================================

    private List<NamedOnnxValue> BuildDecoderInputs(
        float[] embeds,
        int seqLen,
        long[] maskArr,
        long[] posArr,
        int totalSeq,
        HybridState state,
        int pastLen)
    {
        var inputs = new List<NamedOnnxValue>();

        inputs.Add(NamedOnnxValue.CreateFromTensor("inputs_embeds",
            new DenseTensor<float>(
                embeds.AsMemory(0, seqLen * _cfg.HiddenSize),
                [1, seqLen, _cfg.HiddenSize])));

        inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask",
            new DenseTensor<long>(maskArr.AsMemory(0, totalSeq), [1, totalSeq])));

        inputs.Add(NamedOnnxValue.CreateFromTensor("position_ids",
            new DenseTensor<long>(posArr.AsMemory(0, 3 * seqLen), [3, 1, seqLen])));

        for (int L = 0; L < _cfg.NumHiddenLayers; L++)
        {
            bool isFull = _cfg.FullAttentionLayers.Contains(L);

            if (!isFull)
            {
                int convSize = _convStateSize * _convKernelDim;
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_conv.{L}",
                    new DenseTensor<Float16>(
                        state.ConvStates[L].AsMemory(0, convSize),
                        [1, _convStateSize, _convKernelDim])));

                int recSize = _recurrentNH * _recurrentKD * _recurrentKD;
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_recurrent.{L}",
                    new DenseTensor<Float16>(
                        state.RecurrentStates[L].AsMemory(0, recSize),
                        [1, _recurrentNH, _recurrentKD, _recurrentKD])));
            }
            else
            {
                var kv = state.KVCache[L];
                int kvSz = _cfg.NumKeyValueHeads * pastLen * _cfg.HeadDim;
                int[] kvSh = [1, _cfg.NumKeyValueHeads, pastLen, _cfg.HeadDim];

                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{L}.key",
                    kv.KeyLen == 0
                        ? new DenseTensor<Float16>([1, _cfg.NumKeyValueHeads, 0, _cfg.HeadDim])
                        : new DenseTensor<Float16>(kv.Key!.AsMemory(0, kvSz), kvSh)));

                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{L}.value",
                    kv.KeyLen == 0
                        ? new DenseTensor<Float16>([1, _cfg.NumKeyValueHeads, 0, _cfg.HeadDim])
                        : new DenseTensor<Float16>(kv.Val!.AsMemory(0, kvSz), kvSh)));
            }
        }

        return inputs;
    }

    // =========================================================================
    //  UPDATE STATE FROM OUTPUTS (all float16)
    // =========================================================================

    private void UpdateState(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        HybridState state,
        int totalSeq)
    {
        int outIdx = 1;

        for (int L = 0; L < _cfg.NumHiddenLayers; L++)
        {
            bool isFull = _cfg.FullAttentionLayers.Contains(L);

            if (!isFull)
            {
                int convSize = _convStateSize * _convKernelDim;
                var ct = results[outIdx++].AsTensor<Float16>();
                if (ct is DenseTensor<Float16> dct)
                    dct.Buffer.Span.Slice(0, convSize)
                        .CopyTo(state.ConvStates[L].AsSpan(0, convSize));
                else
                    for (int i = 0; i < convSize; i++)
                        state.ConvStates[L][i] = ct.GetValue(i);

                int recSize = _recurrentNH * _recurrentKD * _recurrentKD;
                var rt = results[outIdx++].AsTensor<Float16>();
                if (rt is DenseTensor<Float16> drt)
                    drt.Buffer.Span.Slice(0, recSize)
                        .CopyTo(state.RecurrentStates[L].AsSpan(0, recSize));
                else
                    for (int i = 0; i < recSize; i++)
                        state.RecurrentStates[L][i] = rt.GetValue(i);
            }
            else
            {
                int newSz = _cfg.NumKeyValueHeads * totalSeq * _cfg.HeadDim;
                var keyT = results[outIdx++].AsTensor<Float16>();
                var valT = results[outIdx++].AsTensor<Float16>();

                state.KVCache[L].ReturnToPool();
                var keyBuf = ArrayPool<Float16>.Shared.Rent(newSz);
                var valBuf = ArrayPool<Float16>.Shared.Rent(newSz);

                if (keyT is DenseTensor<Float16> dk && valT is DenseTensor<Float16> dv)
                {
                    dk.Buffer.Span.Slice(0, newSz).CopyTo(keyBuf.AsSpan(0, newSz));
                    dv.Buffer.Span.Slice(0, newSz).CopyTo(valBuf.AsSpan(0, newSz));
                }
                else
                {
                    for (int i = 0; i < newSz; i++)
                    {
                        keyBuf[i] = keyT.GetValue(i);
                        valBuf[i] = valT.GetValue(i);
                    }
                }

                state.KVCache[L] = new KVEntryF16(keyBuf, valBuf, totalSeq);
            }
        }
    }

    // =========================================================================
    //  HELPERS
    // =========================================================================

    private void InspectSessionInputs()
    {
        foreach (var inp in _decoderSession.InputMetadata)
        {
            if (inp.Key.StartsWith("past_conv."))
            {
                var d = inp.Value.Dimensions;
                if (d.Length >= 3) { _convStateSize = d[1]; _convKernelDim = d[2]; }
            }
            else if (inp.Key.StartsWith("past_recurrent."))
            {
                var d = inp.Value.Dimensions;
                if (d.Length >= 4) { _recurrentNH = d[1]; _recurrentKD = d[2]; }
            }
        }
        _logger.LogInformation(
            "[Qwen35] ConvState=[{CS},{CK}] Recurrent=[{RN},{RK}]",
            _convStateSize, _convKernelDim, _recurrentNH, _recurrentKD);
    }

    private static string[] BuildOutputNames(ModelConfig cfg)
    {
        var names = new List<string> { "logits" };
        for (int L = 0; L < cfg.NumHiddenLayers; L++)
        {
            if (!cfg.FullAttentionLayers.Contains(L))
            {
                names.Add($"present_conv.{L}");
                names.Add($"present_recurrent.{L}");
            }
            else
            {
                names.Add($"present.{L}.key");
                names.Add($"present.{L}.value");
            }
        }
        return names.ToArray();
    }

    private static Microsoft.ML.OnnxRuntime.SessionOptions BuildSessionOptions(ModelConfig cfg)
    {
        var opts = new Microsoft.ML.OnnxRuntime.SessionOptions();

        // ── Thread config ────────────────────────────────────────────────────
        opts.IntraOpNumThreads = cfg.IntraOpThreads;  // 12 for 16 vCPU VM
        opts.InterOpNumThreads = cfg.InterOpThreads;  // 2

        // ── Execution mode ───────────────────────────────────────────────────
        // ORT_PARALLEL: allows parallel execution of independent graph nodes.
        // Good for Qwen3.5 which has parallel attention branches.
        opts.ExecutionMode = ExecutionMode.ORT_PARALLEL;

        // ── Graph optimization ───────────────────────────────────────────────
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        // ── Memory optimization ──────────────────────────────────────────────
        opts.EnableCpuMemArena = true;   // BFC arena — reduces alloc overhead
        opts.EnableMemoryPattern = true;   // reuse buffers across runs
        opts.EnableProfiling = false;

        // ── CPU spinning ─────────────────────────────────────────────────────
        // Allow spinning on Windows VM:
        //   - Pros: eliminates thread wake-up latency between decode steps (~0.1ms/step saved)
        //   - Cons: burns CPU cycles when idle — on a SHARED host this means
        //     the hypervisor sees high CPU utilization even between requests,
        //     which can affect vCPU scheduling priority.
        //
        // Recommendation: keep spinning ENABLED if:
        //   - You have dedicated users / continuous load
        //   - Response latency matters more than idle CPU cost
        //
        // Set to "0" if:
        //   - Server is idle most of the time (cost optimization)
        //   - Hosting provider throttles based on CPU utilization
        opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
        opts.AddSessionConfigEntry("session.inter_op.allow_spinning", "1");

        // ── Model bytes optimization ─────────────────────────────────────────
        opts.AddSessionConfigEntry("session.use_ort_model_bytes_directly", "1");

        // ── Prepacking ───────────────────────────────────────────────────────
        // Keep prepacking enabled (0 = don't disable) — prepacks weight matrices
        // once at session init into optimal layout for DNNL/MKL kernels.
        // This is a one-time cost at startup that pays off across all inferences.
        opts.AddSessionConfigEntry("session.disable_prepacking", "0");

        // ── Windows-specific: Force sequential memory allocation ──────────────
        // On Windows VM, the BFC arena can fragment virtual address space.
        // This hint helps the arena allocate contiguously.
        opts.AddSessionConfigEntry("session.use_env_allocators", "1");

        // ── Dynamic block base: reduces latency variance on shared VM host ────
        //
        // ORT official recommendation for environments where threads can be
        // preempted mid-execution (exactly what happens on shared VMware hosts
        // where the hypervisor migrates vCPUs between physical cores).
        //
        // How it works: ORT thread pool divides computation tasks into blocks.
        // With dynamic_block_base=4, the block granularity decreases dynamically
        // as work is claimed — threads grab increasingly smaller chunks of work.
        // This means:
        //   - If a vCPU gets preempted mid-block, the remaining work gets stolen
        //     by other threads faster (smaller blocks = finer-grained work stealing)
        //   - Better load balance across 12 intra-op threads when host is busy
        //   - Lower latency variance (reduces P99 spikes during long prefill)
        //   - Slight throughput improvement for long sequences (64K ctx)
        //
        // Value=4 is the ORT team's recommended starting point.
        // Higher values (8, 16) = finer granularity but more scheduling overhead.
        // Lower values (1, 2) = coarser granularity, less effective on preemptive hosts.
        opts.AddSessionConfigEntry("session.dynamic_block_base", "4");

        return opts;
    }

    public void Dispose()
    {
        _embedSession.Dispose();
        _decoderSession.Dispose();
    }
}

// ============================================================
//  HybridState — Float16 state storage
// ============================================================

internal sealed class HybridState : IDisposable
{
    public readonly Float16[][] ConvStates;
    public readonly Float16[][] RecurrentStates;
    public KVEntryF16[] KVCache;

    public HybridState(ModelConfig cfg,
                       int convStateSize, int convKernelDim,
                       int recurrentNH, int recurrentKD)
    {
        int N = cfg.NumHiddenLayers;
        var fullSet = cfg.FullAttentionLayers.ToHashSet();

        ConvStates = new Float16[N][];
        RecurrentStates = new Float16[N][];
        KVCache = new KVEntryF16[N];

        for (int L = 0; L < N; L++)
        {
            if (!fullSet.Contains(L))
            {
                ConvStates[L] = new Float16[convStateSize * convKernelDim];
                RecurrentStates[L] = new Float16[recurrentNH * recurrentKD * recurrentKD];
                KVCache[L] = new KVEntryF16(null, null, 0);
            }
            else
            {
                ConvStates[L] = Array.Empty<Float16>();
                RecurrentStates[L] = Array.Empty<Float16>();
                KVCache[L] = new KVEntryF16(null, null, 0);
            }
        }
    }

    public void Dispose()
    {
        for (int L = 0; L < KVCache.Length; L++)
            KVCache[L].ReturnToPool();
    }
}

// ============================================================
//  KVEntryF16
// ============================================================

internal struct KVEntryF16
{
    public Float16[]? Key;
    public Float16[]? Val;
    public int KeyLen;

    public KVEntryF16(Float16[]? key, Float16[]? val, int keyLen)
    {
        Key = key;
        Val = val;
        KeyLen = keyLen;
    }

    public void ReturnToPool()
    {
        if (Key is not null) { ArrayPool<Float16>.Shared.Return(Key); Key = null; }
        if (Val is not null) { ArrayPool<Float16>.Shared.Return(Val); Val = null; }
        KeyLen = 0;
    }
}