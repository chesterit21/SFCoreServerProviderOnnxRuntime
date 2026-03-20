// ============================================================
//  Program.cs — SFCore ONNX Runtime API Server
//
//  Optimized for: VMware shared-host Windows VPS, 16 vCPU, 64 GB RAM
//
//  Changelog vs previous:
//  - FIXED: MaxRequestBodySize 64KB → 10MB (was rejecting >64KB payloads)
//  - FIXED: ThreadPool.SetMinThreads 18 → 24 (starvation warning in logs)
//  - ADDED: SemaphoreSlim concurrency guard (1 inference at a time)
//  - ADDED: Per-request inference timeout (default 20 min, configurable)
//  - ADDED: Prompt token limit check with early reject + warning log
//  - ADDED: inferenceSlot field in /health response
//
//  Fully OpenAI-Compatible. Auto-detects model family.
//  Supports: Qwen3 (standard), Qwen3.5 (hybrid linear+full attn)
//
//  Endpoints:
//    GET  /health                 → server status (no auth)
//    GET  /v1/models              → list loaded model
//    POST /v1/chat/completions    → OpenAI chat (stream & non-stream)
//    POST /v1/generate            → raw SSE (custom, internal)
// ============================================================

using Serilog;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

Log.Logger = new LoggerConfiguration()
    .ReadFrom.Configuration(builder.Configuration)
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
    .CreateLogger();
builder.Host.UseSerilog();

// ── ThreadPool tuning for Windows VM ─────────────────────────────────────────
//
//  Problem observed in logs:
//  [10:55:45 WRN] heartbeat running for 00:00:01.16 which is longer than 00:00:01.
//                 This could be caused by thread pool starvation.
//
//  Root cause: a 32K-token prefill blocks 12 ORT threads for ~13 minutes.
//  The .NET ThreadPool's hill-climbing algorithm sees all threads busy and
//  slowly injects new threads (~one per 500ms), but Kestrel's heartbeat
//  fires before new threads are available → starvation warning.
//
//  Fix: Set minimum threads high enough that Kestrel + GC + timers always
//  have threads available even when ORT is fully saturated.
//
//  Budget:
//    12 → ORT intra-op threads (saturated during inference)
//     4 → Kestrel request handling + heartbeat + internal timers
//     4 → .NET server GC threads
//     4 → buffer for async continuations, background work
//  Total minimum: 24 worker threads
ThreadPool.SetMinThreads(workerThreads: 24, completionPortThreads: 8);

builder.Services.ConfigureHttpJsonOptions(opts =>
{
    opts.SerializerOptions.PropertyNameCaseInsensitive = true;
    opts.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    opts.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});

builder.Services.AddSingleton<OnnxModelManager>();
builder.Services.AddHostedService<ModelStartupLoader>();
builder.Services.AddSingleton<ApiKeyService>();

// ── Inference concurrency guard ───────────────────────────────────────────────
// CPU inference is NOT parallelizable across requests — running two ORT
// sessions simultaneously causes thread contention and makes BOTH slower.
// This semaphore ensures only ONE inference runs at a time on the CPU.
// Other requests wait (up to QueueTimeoutSeconds) then receive 503.
builder.Services.AddSingleton<SemaphoreSlim>(_ => new SemaphoreSlim(1, 1));

builder.WebHost.ConfigureKestrel(k =>
{
    // ── FIXED: was 65536 (64KB) — was rejecting any payload > 64KB ───────────
    // A 10K token request has a payload of ~50KB.
    // A 32K token request has a payload of ~120-180KB.
    // Previous value of 64KB was silently rejecting all large context requests.
    // Set to 10MB — far more than any real use case, prevents OOM from abuse.
    k.Limits.MaxRequestBodySize = 10 * 1024 * 1024; // 10 MB

    // Response buffer for SSE streaming
    k.Limits.MaxResponseBufferSize = 65536;

    // Keep-alive: SSE holds open for full inference duration.
    // 64K tokens @ ~40 tok/s prefill = ~27 min prefill alone.
    // Set 60 min to give full headroom for prefill + decode.
    k.Limits.KeepAliveTimeout = TimeSpan.FromMinutes(60);

    // Request headers timeout: protect against slow-header attacks
    k.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(15);
});

var app = builder.Build();

// ── Runtime config (override in appsettings.json if needed) ──────────────────
int queueTimeoutSeconds = app.Configuration.GetValue("InferenceSettings:QueueTimeoutSeconds", 30);
int inferenceTimeoutMinutes = app.Configuration.GetValue("InferenceSettings:InferenceTimeoutMinutes", 60);
int maxPromptTokens = app.Configuration.GetValue("InferenceSettings:MaxPromptTokens", 65536);

// ── MIDDLEWARE: API Key Auth ──────────────────────────────────────────────────
app.Use(async (ctx, next) =>
{
    if (ctx.Request.Path.StartsWithSegments("/health")) { await next(ctx); return; }

    var keyService = ctx.RequestServices.GetRequiredService<ApiKeyService>();
    if (!ctx.Request.Headers.TryGetValue("Authorization", out var auth) ||
        !auth.ToString().StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
    {
        await WriteOpenAIError(ctx, 401, "missing_api_key",
            "Missing or invalid Authorization header. Expected: Bearer <api-key>",
            "invalid_request_error");
        return;
    }

    var key = auth.ToString()["Bearer ".Length..].Trim();
    if (!keyService.IsValid(key))
    {
        Log.Warning("[Auth] Invalid key from {IP}", ctx.Connection.RemoteIpAddress);
        await WriteOpenAIError(ctx, 401, "invalid_api_key",
            "Incorrect API key provided.", "authentication_error");
        return;
    }
    await next(ctx);
});

// ── GET /health ───────────────────────────────────────────────────────────────
app.MapGet("/health", (OnnxModelManager m, SemaphoreSlim sem) => Results.Json(new
{
    status = m.IsReady ? "healthy" : "loading",
    model = m.Config?.ModelName ?? "loading",
    family = m.Config?.Family.ToString() ?? "unknown",
    memoryMB = m.MemoryUsageMB,
    cores = Environment.ProcessorCount,
    threads = m.Config?.IntraOpThreads ?? 0,
    platform = "windows-vm",
    inferenceSlot = sem.CurrentCount > 0 ? "free" : "busy"
}));

// ── GET /v1/models ────────────────────────────────────────────────────────────
app.MapGet("/v1/models", (OnnxModelManager m) =>
{
    var modelId = m.Config?.ModelName ?? Path.GetFileName(
        (app.Configuration["ModelSettings:ModelPath"] ?? "unknown").TrimEnd('/'));
    return Results.Json(new
    {
        @object = "list",
        data = new[]
        {
            new
            {
                id         = modelId,
                @object    = "model",
                created    = 1700000000,
                owned_by   = "sfcore",
                permission = Array.Empty<object>()
            }
        }
    });
});

// ── GET /v1/system/status ─────────────────────────────────────────────────────
// Detailed runtime status: memory, inference slot, config limits.
// Useful for monitoring without digging through logs.
app.MapGet("/v1/system/status", (OnnxModelManager m, SemaphoreSlim sem) =>
{
    var gcInfo = GC.GetGCMemoryInfo();
    long usedBytes = GC.GetTotalMemory(false);
    long totalAvail = gcInfo.TotalAvailableMemoryBytes;
    long heapCommit = gcInfo.TotalCommittedBytes;

    return Results.Json(new
    {
        model = new
        {
            loaded = m.IsReady,
            name = m.Config?.ModelName ?? "not loaded",
            family = m.Config?.Family.ToString() ?? "unknown",
            ram_mb = m.MemoryUsageMB,
            threads = m.Config?.IntraOpThreads ?? 0
        },
        inference = new
        {
            slot = sem.CurrentCount > 0 ? "free" : "busy",
            max_prompt_tokens = maxPromptTokens,
            timeout_minutes = inferenceTimeoutMinutes,
            queue_timeout_s = queueTimeoutSeconds
        },
        memory = new
        {
            dotnet_heap_used_mb = usedBytes / 1024 / 1024,
            dotnet_heap_commit_mb = heapCommit / 1024 / 1024,
            total_available_mb = totalAvail / 1024 / 1024,
            utilization_pct = totalAvail > 0
                                     ? Math.Round((double)usedBytes / totalAvail * 100, 1)
                                     : 0
        },
        platform = new
        {
            os = "Windows",
            vcpu = Environment.ProcessorCount,
            vm = true,
            host_type = "VMware shared"
        }
    });
});


app.MapPost("/v1/chat/completions", async (HttpContext ctx, OnnxModelManager m, SemaphoreSlim sem) =>
{
    if (!m.IsReady)
    {
        await WriteOpenAIError(ctx, 503, "model_not_ready",
            "Model is still loading, retry in a few seconds.", "server_error");
        return;
    }

    OpenAIChatRequest? req;
    try { req = await ctx.Request.ReadFromJsonAsync<OpenAIChatRequest>(); }
    catch
    {
        await WriteOpenAIError(ctx, 400, "invalid_json",
            "Could not parse request body as JSON.", "invalid_request_error");
        return;
    }

    if (req?.Messages is null || req.Messages.Count == 0)
    {
        await WriteOpenAIError(ctx, 400, "invalid_request",
            "messages array is required and must not be empty.", "invalid_request_error");
        return;
    }

    var modelId = m.Config?.ModelName ?? "sfcore-model";
    bool enableThink = req.EnableThinking ?? false;
    var prompt = BuildChatMLPrompt(req.Messages, enableThink);
    var tokenIds = m.Tokenizer!.Encode(prompt);
    int promptTokens = tokenIds.Length;

    // ── Prompt token limit ────────────────────────────────────────────────────
    // A 32K prompt = ~13 min prefill, blocking the server for all other users.
    // Reject early with a clear error rather than silently blocking everything.
    if (promptTokens > maxPromptTokens)
    {
        Log.Warning("[Chat] Prompt rejected: {T} tokens > max {M}", promptTokens, maxPromptTokens);
        await WriteOpenAIError(ctx, 400, "context_length_exceeded",
            $"Prompt is {promptTokens} tokens which exceeds the maximum of {maxPromptTokens}.",
            "invalid_request_error");
        return;
    }

    // Warn in logs for large prompts so you can see estimated wait time.
    // ~40 tokens/sec prefill confirmed from benchmark logs.
    // 8K  tokens → ~200s  (~3.3 min)
    // 16K tokens → ~400s  (~6.7 min)
    // 32K tokens → ~800s  (~13 min)
    // 64K tokens → ~1600s (~27 min)
    if (promptTokens > 8000)
        Log.Warning("[Chat] Large prompt: {T} tokens — estimated prefill ~{S}s (~{M}min)",
            promptTokens, promptTokens / 40, promptTokens / 40 / 60);

    bool stream = req.Stream ?? false;
    string completionId = $"chatcmpl-{Guid.NewGuid():N}";
    long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    float temperature = req.Temperature ?? m.Config?.DefaultTemperature ?? 0.7f;
    float topP = req.TopP ?? m.Config?.DefaultTopP ?? 0.9f;
    int topK = req.TopK ?? m.Config?.DefaultTopK ?? 20;
    int maxTokens = req.MaxTokens ?? 1024;

    // ── Acquire inference slot ────────────────────────────────────────────────
    bool acquired = await sem.WaitAsync(
        TimeSpan.FromSeconds(queueTimeoutSeconds), ctx.RequestAborted);
    if (!acquired)
    {
        Log.Warning("[Chat] Rejected after {T}s queue wait — slot busy (prompt={P} tokens)",
            queueTimeoutSeconds, promptTokens);
        await WriteOpenAIError(ctx, 503, "server_busy",
            "Server is processing another request. Please retry in a moment.",
            "server_error");
        return;
    }

    // Inference timeout: prevents runaway long requests from blocking forever
    using var timeoutCts = new CancellationTokenSource(
        TimeSpan.FromMinutes(inferenceTimeoutMinutes));
    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
        ctx.RequestAborted, timeoutCts.Token);

    try
    {
        // ── STREAMING ────────────────────────────────────────────────────────
        if (stream)
        {
            ctx.Response.Headers["Content-Type"] = "text/event-stream";
            ctx.Response.Headers["Cache-Control"] = "no-cache";
            ctx.Response.Headers["X-Accel-Buffering"] = "no";
            ctx.Response.Headers["Connection"] = "keep-alive";

            var firstChunk = BuildStreamChunk(completionId, created, modelId,
                deltaRole: "assistant", deltaContent: null, finishReason: null);
            await ctx.Response.WriteAsync($"data: {firstChunk}\n\n");
            await ctx.Response.Body.FlushAsync();

            int completionTokens = 0;
            bool thinkDone = false;
            var thinkBuf = new StringBuilder(256);
            var streamDec = m.Tokenizer!.CreateStreamingDecoder();

            try
            {
                await foreach (var tokenId in m.GenerateAsync(
                    tokenIds, maxTokens, temperature, topP, topK,
                    linkedCts.Token))
                {
                    var rawToken = m.Tokenizer.GetTokenString(tokenId);
                    var text = streamDec.Decode(rawToken);
                    if (string.IsNullOrEmpty(text)) continue;

                    if (!thinkDone)
                    {
                        thinkBuf.Append(text);
                        var buf = thinkBuf.ToString();
                        if (buf.Contains("</think>"))
                        {
                            thinkDone = true;
                            text = buf[(buf.IndexOf("</think>") + "</think>".Length)..]
                                .TrimStart('\n', '\r', ' ');
                            if (string.IsNullOrEmpty(text)) continue;
                        }
                        else if (!buf.Contains("<think>") && buf.Length > 30)
                        {
                            thinkDone = true;
                            text = buf.TrimStart('\n', '\r', ' ');
                            if (string.IsNullOrEmpty(text)) continue;
                        }
                        else continue;
                    }

                    completionTokens++;
                    var chunk = BuildStreamChunk(completionId, created, modelId, null, text, null);
                    await ctx.Response.WriteAsync($"data: {chunk}\n\n");
                    await ctx.Response.Body.FlushAsync();
                }

                var flushed = streamDec.Flush();
                if (!string.IsNullOrEmpty(flushed))
                {
                    var fc = BuildStreamChunk(completionId, created, modelId, null, flushed, null);
                    await ctx.Response.WriteAsync($"data: {fc}\n\n");
                    await ctx.Response.Body.FlushAsync();
                }

                var finalChunk = BuildStreamChunk(completionId, created, modelId, null, null, "stop");
                await ctx.Response.WriteAsync($"data: {finalChunk}\n\n");
                await ctx.Response.WriteAsync("data: [DONE]\n\n");
                await ctx.Response.Body.FlushAsync();

                Log.Information("[Chat] stream done — prompt={P} completion={C} tokens",
                    promptTokens, completionTokens);
            }
            catch (OperationCanceledException)
            {
                if (timeoutCts.IsCancellationRequested)
                    Log.Warning("[Chat] Inference timeout ({M}min) — prompt={P} tokens",
                        inferenceTimeoutMinutes, promptTokens);
                try
                {
                    await ctx.Response.WriteAsync(
                        $"data: {BuildStreamChunk(completionId, created, modelId, null, null, "stop")}\n\n");
                    await ctx.Response.WriteAsync("data: [DONE]\n\n");
                    await ctx.Response.Body.FlushAsync();
                }
                catch { }
            }
            catch (Exception ex)
            {
                Log.Error(ex, "[Chat] stream error");
                try
                {
                    await ctx.Response.WriteAsync(
                        $"data: {BuildStreamChunk(completionId, created, modelId, null, null, "error")}\n\n");
                    await ctx.Response.WriteAsync("data: [DONE]\n\n");
                    await ctx.Response.Body.FlushAsync();
                }
                catch { }
            }
            return;
        }

        // ── NON-STREAMING ─────────────────────────────────────────────────────
        var sb = new StringBuilder();
        int compTokens = 0;
        bool tdone = false;
        var tbuf = new StringBuilder(256);

        try
        {
            await foreach (var tokenId in m.GenerateAsync(
                tokenIds, maxTokens, temperature, topP, topK,
                linkedCts.Token))
            {
                var text = m.Tokenizer!.DecodeSingle(tokenId);
                if (!tdone)
                {
                    tbuf.Append(text);
                    var buf = tbuf.ToString();
                    if (buf.Contains("</think>"))
                    {
                        tdone = true;
                        text = buf[(buf.IndexOf("</think>") + "</think>".Length)..]
                            .TrimStart('\n', '\r', ' ');
                        if (string.IsNullOrEmpty(text)) continue;
                    }
                    else if (!buf.Contains("<think>") && buf.Length > 30)
                    {
                        tdone = true;
                        text = buf.TrimStart('\n', '\r', ' ');
                        if (string.IsNullOrEmpty(text)) continue;
                    }
                    else continue;
                }
                sb.Append(text);
                compTokens++;
            }
        }
        catch (OperationCanceledException)
        {
            if (timeoutCts.IsCancellationRequested)
                Log.Warning("[Chat] Non-stream timeout ({M}min) — prompt={P} tokens",
                    inferenceTimeoutMinutes, promptTokens);
        }

        var respOpts = new JsonSerializerOptions
        {
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
        };
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(JsonSerializer.Serialize(new
        {
            id = completionId,
            @object = "chat.completion",
            created,
            model = modelId,
            choices = new[]
            {
                new
                {
                    index         = 0,
                    message       = new { role = "assistant", content = sb.ToString() },
                    finish_reason = "stop"
                }
            },
            usage = new
            {
                prompt_tokens = promptTokens,
                completion_tokens = compTokens,
                total_tokens = promptTokens + compTokens
            }
        }, respOpts));

        Log.Information("[Chat] done — prompt={P} completion={C} tokens", promptTokens, compTokens);
    }
    finally
    {
        // Always release — even if inference threw an exception
        sem.Release();
    }
});

// ── POST /v1/generate — raw SSE ───────────────────────────────────────────────
app.MapPost("/v1/generate", async (HttpContext ctx, OnnxModelManager m, SemaphoreSlim sem) =>
{
    if (!m.IsReady)
    {
        ctx.Response.StatusCode = 503;
        await ctx.Response.WriteAsJsonAsync(new { error = "Model loading" });
        return;
    }

    GenerateRequest? req;
    try { req = await ctx.Request.ReadFromJsonAsync<GenerateRequest>(); }
    catch
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "Invalid JSON" });
        return;
    }

    if (req is null || string.IsNullOrWhiteSpace(req.Prompt))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "prompt is required" });
        return;
    }

    bool acquired = await sem.WaitAsync(TimeSpan.FromSeconds(30), ctx.RequestAborted);
    if (!acquired)
    {
        ctx.Response.StatusCode = 503;
        await ctx.Response.WriteAsJsonAsync(new { error = "Server busy, retry shortly" });
        return;
    }

    ctx.Response.Headers["Content-Type"] = "text/event-stream";
    ctx.Response.Headers["Cache-Control"] = "no-cache";
    ctx.Response.Headers["X-Accel-Buffering"] = "no";
    ctx.Response.Headers["Connection"] = "keep-alive";

    var tokenIds = m.Tokenizer!.Encode(req.Prompt);
    int tokenCount = 0;

    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromMinutes(60));
    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
        ctx.RequestAborted, timeoutCts.Token);

    try
    {
        await foreach (var tokenId in m.GenerateAsync(
            tokenIds,
            req.MaxTokens > 0 ? req.MaxTokens : 512,
            req.Temperature > 0f ? req.Temperature : 0.7f,
            req.TopP > 0f ? req.TopP : 0.9f,
            req.TopK > 0 ? req.TopK : 20,
            linkedCts.Token))
        {
            tokenCount++;
            var text = m.Tokenizer.DecodeSingle(tokenId);
            var json = JsonSerializer.Serialize(new
            {
                text,
                token_id = tokenId,
                tokens_used = tokenCount,
                is_final = false
            });
            await ctx.Response.WriteAsync($"data: {json}\n\n");
            await ctx.Response.Body.FlushAsync();
        }

        var final = JsonSerializer.Serialize(new
        {
            text = "",
            tokens_used = tokenCount,
            is_final = true,
            finish_reason = "stop"
        });
        await ctx.Response.WriteAsync($"data: {final}\n\n");
        await ctx.Response.Body.FlushAsync();
    }
    catch (OperationCanceledException) { }
    catch (Exception ex)
    {
        Log.Error(ex, "[Generate] error");
        try
        {
            await ctx.Response.WriteAsync(
                $"data: {JsonSerializer.Serialize(new { error = ex.Message, is_final = true })}\n\n");
            await ctx.Response.Body.FlushAsync();
        }
        catch { }
    }
    finally
    {
        sem.Release();
    }
});

Log.Information(
    "SFCore ONNX Server starting — {Cores} vCPU | Windows VM | ORT threads configured",
    Environment.ProcessorCount);

app.Run();

// ── HELPERS ──────────────────────────────────────────────────────────────────

static string BuildChatMLPrompt(List<OAIMessage> messages, bool enableThinking)
{
    var sb = new StringBuilder();
    foreach (var msg in messages)
        sb.Append($"<|im_start|>{msg.Role}\n{msg.Content}<|im_end|>\n");
    sb.Append("<|im_start|>assistant\n");
    if (!enableThinking)
        sb.Append("<think>\n\n</think>\n\n");
    return sb.ToString();
}

static string BuildStreamChunk(
    string completionId, long created, string model,
    string? deltaRole, string? deltaContent, string? finishReason)
{
    var enc = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping;
    var opts = new JsonSerializerOptions { Encoder = enc };

    string deltaJson = (deltaRole, deltaContent) switch
    {
        (not null, _) => $"{{\"role\":\"{deltaRole}\"}}",
        (_, not null) => $"{{\"content\":{JsonSerializer.Serialize(deltaContent, opts)}}}",
        _ => "{}"
    };

    string finishJson = finishReason is null ? "null" : $"\"{finishReason}\"";
    return $"{{\"id\":{JsonSerializer.Serialize(completionId, opts)}," +
           $"\"object\":\"chat.completion.chunk\"," +
           $"\"created\":{created}," +
           $"\"model\":{JsonSerializer.Serialize(model, opts)}," +
           $"\"choices\":[{{\"index\":0,\"delta\":{deltaJson},\"finish_reason\":{finishJson}}}]}}";
}

static async Task WriteOpenAIError(HttpContext ctx, int status, string code,
    string message, string type)
{
    ctx.Response.StatusCode = status;
    ctx.Response.ContentType = "application/json";
    var opts = new JsonSerializerOptions
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };
    await ctx.Response.WriteAsync(JsonSerializer.Serialize(
        new { error = new { message, type, code } }, opts));
}

// ── RECORD TYPES ─────────────────────────────────────────────────────────────

public record OAIMessage(string Role, string Content);

public class OpenAIChatRequest
{
    public List<OAIMessage> Messages { get; set; } = [];
    public int? MaxTokens { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public int? TopK { get; set; }
    public bool? Stream { get; set; }
    public string? Model { get; set; }
    [JsonPropertyName("enable_thinking")]
    public bool? EnableThinking { get; set; }
}

public record GenerateRequest(
    string Prompt,
    int MaxTokens = 512,
    float Temperature = 0.7f,
    float TopP = 0.9f,
    int TopK = 20);

public class ApiKeyService
{
    private readonly HashSet<string> _validKeys;
    private readonly ILogger<ApiKeyService> _logger;

    public ApiKeyService(IConfiguration config, ILogger<ApiKeyService> logger)
    {
        _logger = logger;
        var keys = config.GetSection("ApiAuth:Keys").Get<string[]>() ?? [];
        _validKeys = new HashSet<string>(keys, StringComparer.Ordinal);

        if (_validKeys.Count == 0)
            _logger.LogWarning("[Auth] ⚠️  No API keys configured — all requests will be rejected!");
        else
            _logger.LogInformation("[Auth] {Count} API key(s) loaded", _validKeys.Count);
    }

    public bool IsValid(string key) =>
        !string.IsNullOrEmpty(key) && _validKeys.Contains(key);
}