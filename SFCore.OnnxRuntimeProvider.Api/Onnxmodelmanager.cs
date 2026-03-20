// ============================================================
//  OnnxModelManager.cs — Unified model manager
//  Supports Qwen3 (standard) and Qwen3.5 (hybrid)
// ============================================================

public class OnnxModelManager : IDisposable
{
    private readonly ILogger<OnnxModelManager> _logger;
    private readonly string _modelPath;

    public ModelConfig? Config { get; private set; }
    public QwenTokenizer? Tokenizer { get; private set; }

    public QwenInferenceEngine? EngineQwen3 { get; private set; }
    public Qwen35InferenceEngine? EngineQwen35 { get; private set; }

    public bool IsReady { get; private set; }
    public long MemoryUsageMB => GC.GetTotalMemory(false) / 1024 / 1024;

    public OnnxModelManager(ILogger<OnnxModelManager> logger, IConfiguration config)
    {
        _logger = logger;
        var rel = config["ModelSettings:ModelPath"] ?? "models";
        _modelPath = Path.IsPathRooted(rel) ? rel : Path.Combine(AppContext.BaseDirectory, rel);
    }

    public void Load()
    {
        if (IsReady) return;

        _logger.LogInformation("[Manager] Load() entered. Path: {P}", _modelPath);

        // ── Step 1: Hardware ─────────────────────────────────────────────────
        _logger.LogInformation("[Manager] Step 1: Hardware detection...");
        HardwareDetector.HardwareInfo hw;
        try { hw = HardwareDetector.Detect(_logger); }
        catch (Exception ex) { _logger.LogCritical(ex, "[Manager] CRASH at Step 1: Hardware"); throw; }

        // ── Step 2: ModelConfig ──────────────────────────────────────────────
        _logger.LogInformation("[Manager] Step 2: ModelConfig.Load()...");
        try { Config = ModelConfig.Load(_modelPath, _logger); }
        catch (Exception ex) { _logger.LogCritical(ex, "[Manager] CRASH at Step 2: ModelConfig"); throw; }

        _logger.LogInformation("[Manager] Step 2b: ApplyToConfig...");
        try { HardwareDetector.ApplyToConfig(Config, hw, _logger); }
        catch (Exception ex) { _logger.LogCritical(ex, "[Manager] CRASH at Step 2b: ApplyToConfig"); throw; }

        // ── Step 3: Tokenizer ────────────────────────────────────────────────
        _logger.LogInformation("[Manager] Step 3: Tokenizer...");
        var tokPath = Path.Combine(_modelPath, "tokenizer.json");
        try
        {
            Tokenizer = new QwenTokenizer(tokPath);
            _logger.LogInformation("[Manager] Vocab size: {V}", Tokenizer.VocabSize);
        }
        catch (Exception ex) { _logger.LogCritical(ex, "[Manager] CRASH at Step 3: Tokenizer"); throw; }

        _logger.LogInformation("[Manager] Step 3b: SyncSpecialTokenIds...");
        try { SyncSpecialTokenIds(Tokenizer, Config); }
        catch (Exception ex) { _logger.LogCritical(ex, "[Manager] CRASH at Step 3b: SyncTokenIds"); throw; }

        // ── Step 4: Engine ───────────────────────────────────────────────────
        _logger.LogInformation("[Manager] Step 4: Engine init. Family={F}", Config.Family);
        try
        {
            switch (Config.Family)
            {
                case ModelFamily.Qwen3_5:
                    _logger.LogInformation("[Manager] Instantiating Qwen35InferenceEngine...");
                    EngineQwen35 = new Qwen35InferenceEngine(Config, _logger);
                    break;
                case ModelFamily.Qwen3:
                default:
                    _logger.LogInformation("[Manager] Instantiating QwenInferenceEngine (Qwen3)...");
                    EngineQwen3 = new QwenInferenceEngine(_modelPath, _logger);
                    break;
            }
        }
        catch (Exception ex) { _logger.LogCritical(ex, "[Manager] CRASH at Step 4: Engine"); throw; }

        IsReady = true;
        GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
        _logger.LogInformation("✅ {Name} ready — RAM: {MB}MB", Config.ModelName, MemoryUsageMB);
    }

    /// <summary>Generate tokens — dispatches to correct engine.</summary>
    public IAsyncEnumerable<int> GenerateAsync(
        int[] inputIds,
        int maxNewTokens = 512,
        float temperature = 0.7f,
        float topP = 0.9f,
        int topK = 20,
        CancellationToken cancellationToken = default)
    {
        if (!IsReady) throw new InvalidOperationException("Model not loaded");

        return Config!.Family switch
        {
            ModelFamily.Qwen3_5 => EngineQwen35!.GenerateAsync(
                inputIds, maxNewTokens, temperature, topP, topK,
                Config.EosTokenIds, cancellationToken),

            _ => EngineQwen3!.GenerateAsync(
                inputIds, maxNewTokens, temperature, topP, topK,
                Config.EosTokenIds, cancellationToken),
        };
    }

    // =========================================================================

    private static void SyncSpecialTokenIds(QwenTokenizer tokenizer, ModelConfig config)
    {
        var imStart = tokenizer.GetSpecialTokenId("<|im_start|>");
        var imEnd = tokenizer.GetSpecialTokenId("<|im_end|>");

        if (imStart.HasValue) config.ImStartTokenId = imStart.Value;
        if (imEnd.HasValue)
        {
            config.ImEndTokenId = imEnd.Value;
            config.EosTokenIds.Add(imEnd.Value);
        }
    }

    public void Dispose()
    {
        EngineQwen3?.Dispose();
        EngineQwen35?.Dispose();
        GC.SuppressFinalize(this);
    }
}

public class ModelStartupLoader : IHostedService
{
    private readonly OnnxModelManager _m;
    private readonly ILogger<ModelStartupLoader> _logger;

    public ModelStartupLoader(OnnxModelManager m, ILogger<ModelStartupLoader> logger)
    {
        _m = m;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken ct)
    {
        _ = Task.Run(() =>
        {
            try
            {
                _logger.LogInformation("[Startup] Background model load starting...");
                _m.Load();
                _logger.LogInformation("[Startup] Background model load complete.");
            }
            catch (Exception ex)
            {
                _logger.LogCritical(ex, "[Startup] ❌ Model load FAILED");
            }
        });
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken ct) { _m.Dispose(); return Task.CompletedTask; }
}