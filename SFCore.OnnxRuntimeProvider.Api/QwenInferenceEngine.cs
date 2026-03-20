// ============================================================
//  QwenInferenceEngine.cs
//
//  Qwen3 standard transformer engine (single ONNX, input_ids).
//  Used when ModelFamily = Qwen3.
//
//  NOTE: Qwen3.5 hybrid engine is in Qwen35InferenceEngine.cs
//  Shared types (KVEntry, Sampler) live here.
// ============================================================

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;
using System.Runtime.CompilerServices;

public sealed class QwenInferenceEngine : IDisposable
{
    private const int NumLayers = 28;
    private const int NumKVHeads = 8;
    private const int HeadDim = 128;
    private const int VocabSize = 151936;

    private readonly InferenceSession _session;
    private readonly ILogger _logger;
    private static readonly string[] OutputNames = BuildOutputNames();
    private readonly float[] _logitsScratch =
        GC.AllocateArray<float>(VocabSize, pinned: true);
    private static readonly HashSet<int> DefaultStopIds = [151645, 151643];

    public QwenInferenceEngine(string modelDir, ILogger logger)
    {
        _logger = logger;
        var modelFile = LocateModelFile(modelDir);
        _logger.LogInformation("[Engine] Loading {F}", Path.GetFileName(modelFile));
        var opts = BuildSessionOptions();
        _session = new InferenceSession(modelFile, opts);
        _logger.LogInformation("[Engine] Ready");
    }

    public async IAsyncEnumerable<int> GenerateAsync(
        int[] inputIds,
        int maxNewTokens = 512,
        float temperature = 1.0f,
        float topP = 0.9f,
        int topK = 20,
        IEnumerable<int>? stopTokenIds = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var stopIds = stopTokenIds as HashSet<int>
                      ?? (stopTokenIds?.ToHashSet() ?? DefaultStopIds);

        var kvCache = new KVEntry[NumLayers];
        for (int L = 0; L < NumLayers; L++)
            kvCache[L] = new KVEntry(null, null, 0);

        try
        {
            int pastLen = 0;
            int nextToken = RunForward(inputIds, kvCache, ref pastLen,
                                       temperature, topP, topK);
            if (stopIds.Contains(nextToken)) yield break;
            yield return nextToken;

            for (int step = 0; step < maxNewTokens - 1; step++)
            {
                if (cancellationToken.IsCancellationRequested) yield break;
                nextToken = RunForward([nextToken], kvCache, ref pastLen,
                                       temperature, topP, topK);
                if (stopIds.Contains(nextToken)) yield break;
                yield return nextToken;
                await Task.Yield();
            }
        }
        finally
        {
            for (int L = 0; L < NumLayers; L++)
                kvCache[L].ReturnToPool();
        }
    }

    private int RunForward(int[] tokenIds, KVEntry[] kvCache,
                           ref int pastLen, float temperature,
                           float topP, int topK)
    {
        int seqLen = tokenIds.Length;
        int totalSeq = pastLen + seqLen;

        long[] inputArr = ArrayPool<long>.Shared.Rent(seqLen);
        long[] maskArr = ArrayPool<long>.Shared.Rent(totalSeq);
        long[] posArr = ArrayPool<long>.Shared.Rent(seqLen);

        try
        {
            for (int i = 0; i < seqLen; i++) inputArr[i] = tokenIds[i];
            for (int i = 0; i < totalSeq; i++) maskArr[i] = 1L;
            for (int i = 0; i < seqLen; i++) posArr[i] = pastLen + i;

            var inputs = new List<NamedOnnxValue>(3 + NumLayers * 2);
            inputs.Add(NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(inputArr.AsMemory(0, seqLen), [1, seqLen])));
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask",
                new DenseTensor<long>(maskArr.AsMemory(0, totalSeq), [1, totalSeq])));
            inputs.Add(NamedOnnxValue.CreateFromTensor("position_ids",
                new DenseTensor<long>(posArr.AsMemory(0, seqLen), [1, seqLen])));

            for (int L = 0; L < NumLayers; L++)
            {
                var e = kvCache[L];
                int[] sh = [1, NumKVHeads, pastLen, HeadDim];
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{L}.key",
                    e.KeyLen == 0
                        ? new DenseTensor<float>([1, NumKVHeads, 0, HeadDim])
                        : new DenseTensor<float>(
                            e.Key!.AsMemory(0, NumKVHeads * pastLen * HeadDim), sh)));
                inputs.Add(NamedOnnxValue.CreateFromTensor($"past_key_values.{L}.value",
                    e.KeyLen == 0
                        ? new DenseTensor<float>([1, NumKVHeads, 0, HeadDim])
                        : new DenseTensor<float>(
                            e.Val!.AsMemory(0, NumKVHeads * pastLen * HeadDim), sh)));
            }

            using var results = _session.Run(inputs, OutputNames);

            var logitsTensor = results[0].AsTensor<float>();
            int logitsOffset = (seqLen - 1) * VocabSize;
            if (logitsTensor is DenseTensor<float> denseLT)
                denseLT.Buffer.Span.Slice(logitsOffset, VocabSize)
                    .CopyTo(_logitsScratch);
            else
                for (int i = 0; i < VocabSize; i++)
                    _logitsScratch[i] = logitsTensor[0, seqLen - 1, i];

            int newKVSize = NumKVHeads * totalSeq * HeadDim;
            for (int L = 0; L < NumLayers; L++)
            {
                var newKey = results[1 + L * 2].AsTensor<float>();
                var newVal = results[1 + L * 2 + 1].AsTensor<float>();
                kvCache[L].ReturnToPool();
                float[] keyBuf = ArrayPool<float>.Shared.Rent(newKVSize);
                float[] valBuf = ArrayPool<float>.Shared.Rent(newKVSize);
                if (newKey is DenseTensor<float> dk && newVal is DenseTensor<float> dv)
                {
                    dk.Buffer.Span.Slice(0, newKVSize).CopyTo(keyBuf.AsSpan(0, newKVSize));
                    dv.Buffer.Span.Slice(0, newKVSize).CopyTo(valBuf.AsSpan(0, newKVSize));
                }
                else
                {
                    for (int h = 0; h < NumKVHeads; h++)
                        for (int t = 0; t < totalSeq; t++)
                            for (int d = 0; d < HeadDim; d++)
                            {
                                int idx = h * totalSeq * HeadDim + t * HeadDim + d;
                                keyBuf[idx] = newKey[0, h, t, d];
                                valBuf[idx] = newVal[0, h, t, d];
                            }
                }
                kvCache[L] = new KVEntry(keyBuf, valBuf, totalSeq);
            }

            pastLen = totalSeq;
            return Sampler.Sample(_logitsScratch, temperature, topP, topK);
        }
        finally
        {
            ArrayPool<long>.Shared.Return(inputArr);
            ArrayPool<long>.Shared.Return(maskArr);
            ArrayPool<long>.Shared.Return(posArr);
        }
    }

    private static string[] BuildOutputNames()
    {
        var names = new string[1 + NumLayers * 2];
        names[0] = "logits";
        for (int L = 0; L < NumLayers; L++)
        {
            names[1 + L * 2] = $"present.{L}.key";
            names[1 + L * 2 + 1] = $"present.{L}.value";
        }
        return names;
    }

    private static Microsoft.ML.OnnxRuntime.SessionOptions BuildSessionOptions()
    {
        var opts = new Microsoft.ML.OnnxRuntime.SessionOptions();
        int logical = Environment.ProcessorCount;
        int physical = Math.Max(1, logical / 2);
        opts.IntraOpNumThreads = physical;
        opts.InterOpNumThreads = 1;
        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.EnableCpuMemArena = true;
        opts.EnableMemoryPattern = true;
        opts.EnableProfiling = false;
        opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
        return opts;
    }

    private static string LocateModelFile(string dir) =>
        Directory.GetFiles(dir, "model_q4.onnx").FirstOrDefault()
        ?? Directory.GetFiles(dir, "model_q4*.onnx").FirstOrDefault()
        ?? Directory.GetFiles(dir, "model_int4*.onnx").FirstOrDefault()
        ?? Directory.GetFiles(dir, "*.onnx").FirstOrDefault()
        ?? throw new FileNotFoundException("No .onnx model found in: " + dir);

    public void Dispose() => _session.Dispose();
}

// ============================================================
//  KVEntry — float32 KV cache entry (used by QwenInferenceEngine)
// ============================================================

internal struct KVEntry
{
    public float[]? Key;
    public float[]? Val;
    public int KeyLen;

    public KVEntry(float[]? key, float[]? val, int keyLen)
    {
        Key = key;
        Val = val;
        KeyLen = keyLen;
    }

    public void ReturnToPool()
    {
        if (Key is not null) { ArrayPool<float>.Shared.Return(Key); Key = null; }
        if (Val is not null) { ArrayPool<float>.Shared.Return(Val); Val = null; }
        KeyLen = 0;
    }
}

// ============================================================
//  Sampler — shared by both engines
// ============================================================

internal static class Sampler
{
    [ThreadStatic] private static Random? _rng;
    private static Random Rng => _rng ??= new Random();

    public static int Sample(float[] logits, float temperature, float topP, int topK)
    {
        if (temperature <= 0f) return ArgMax(logits);

        int len = logits.Length;
        float invTemp = 1f / temperature;
        float[] probs = ArrayPool<float>.Shared.Rent(len);

        try
        {
            float max = logits[0];
            for (int i = 1; i < len; i++) if (logits[i] > max) max = logits[i];
            float sum = 0f;
            for (int i = 0; i < len; i++)
            {
                probs[i] = MathF.Exp((logits[i] - max) * invTemp);
                sum += probs[i];
            }
            float inv = 1f / sum;
            for (int i = 0; i < len; i++) probs[i] *= inv;

            if (topK > 0 && topK < len)
            {
                float threshold = KthLargestValue(probs, len, topK);
                for (int i = 0; i < len; i++)
                    if (probs[i] < threshold) probs[i] = 0f;
            }

            if (topP < 1f)
            {
                int[] idx = ArrayPool<int>.Shared.Rent(len);
                try
                {
                    for (int i = 0; i < len; i++) idx[i] = i;
                    Array.Sort(idx, 0, len,
                        Comparer<int>.Create((a, b) => probs[b].CompareTo(probs[a])));
                    float cum = 0f; int cut = len - 1;
                    for (int i = 0; i < len; i++)
                    {
                        cum += probs[idx[i]];
                        if (cum >= topP) { cut = i; break; }
                    }
                    for (int i = cut + 1; i < len; i++) probs[idx[i]] = 0f;
                }
                finally { ArrayPool<int>.Shared.Return(idx); }
            }

            sum = 0f;
            for (int i = 0; i < len; i++) sum += probs[i];
            if (sum <= 0f) return ArgMax(logits);

            float r = (float)Rng.NextDouble() * sum, c = 0f;
            for (int i = 0; i < len; i++) { c += probs[i]; if (r <= c) return i; }
            return len - 1;
        }
        finally { ArrayPool<float>.Shared.Return(probs); }
    }

    private static int ArgMax(float[] a)
    {
        int b = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i;
        return b;
    }

    private static float KthLargestValue(float[] src, int len, int k)
    {
        float[] buf = ArrayPool<float>.Shared.Rent(len);
        try
        {
            src.AsSpan(0, len).CopyTo(buf.AsSpan(0, len));
            int left = 0, right = len - 1, target = k - 1;
            while (left < right)
            {
                int m = (left + right) / 2;
                float piv = Median3(buf[left], buf[m], buf[right]);
                int l = left, r = right;
                while (l <= r)
                {
                    while (buf[l] > piv) l++;
                    while (buf[r] < piv) r--;
                    if (l <= r) { (buf[l], buf[r]) = (buf[r], buf[l]); l++; r--; }
                }
                if (target <= r) right = r;
                else if (target >= l) left = l;
                else return buf[target];
            }
            return buf[left];
        }
        finally { ArrayPool<float>.Shared.Return(buf); }
    }

    private static float Median3(float a, float b, float c)
    {
        if (a > b) (a, b) = (b, a);
        if (b > c) (b, c) = (c, b);
        if (a > b) (a, b) = (b, a);
        return b;
    }
}