// ============================================================
//  ModelConfig.cs — Auto-detect model architecture & capabilities
//
//  Reads config.json + generation_config.json + tokenizer_config.json
//  from model directory. Zero hardcoded constants.
//
//  Supports:
//    - Qwen3   : standard transformer, single ONNX, input_ids
//    - Qwen3.5 : hybrid linear+full attention, split ONNX,
//                inputs_embeds via embed_tokens model, mrope position_ids
// ============================================================

using System.Text.Json;
using System.Text.Json.Nodes;

public enum ModelFamily
{
    Qwen3,      // standard transformer — model.onnx, input_ids
    Qwen3_5,    // hybrid linear+full attention — decoder + embed_tokens, inputs_embeds
    Unknown
}

public sealed class ModelConfig
{
    // ── Identity ─────────────────────────────────────────────────────────────
    public string ModelDir { get; private set; } = "";
    public string ModelName { get; private set; } = "";
    public ModelFamily Family { get; private set; } = ModelFamily.Unknown;

    // ── Architecture ─────────────────────────────────────────────────────────
    public int NumHiddenLayers { get; private set; }
    public int NumAttentionHeads { get; private set; }
    public int NumKeyValueHeads { get; private set; }
    public int HeadDim { get; private set; }
    public int HiddenSize { get; private set; }
    public int VocabSize { get; private set; }

    // ── Hybrid-specific (Qwen3.5) ────────────────────────────────────────────
    // Layer index → "full_attention" | "linear_attention"
    public IReadOnlyList<string> LayerTypes { get; private set; } = [];
    // Indices of full-attention layers (have KV cache that grows)
    public IReadOnlyList<int> FullAttentionLayers { get; private set; } = [];
    // Conv state shape: [batch, ConvStateSize, ConvKernelDim]
    public int ConvStateSize { get; private set; }   // 6144
    public int ConvKernelDim { get; private set; }   // 4
    // Recurrent state shape: [batch, RecurrentNumHeads, RecurrentKeyDim, RecurrentKeyDim]
    public int RecurrentNumHeads { get; private set; }   // 16
    public int RecurrentKeyDim { get; private set; }   // 128
    // Whether position_ids has shape [3, batch, seq] (mrope) vs [batch, seq]
    public bool UseMRope { get; private set; }

    // ── ONNX files ───────────────────────────────────────────────────────────
    public string DecoderModelFile { get; private set; } = "";
    public string? EmbedTokensFile { get; private set; }   // null for Qwen3

    // ── Tokenizer ────────────────────────────────────────────────────────────
    public int ImStartTokenId { get; set; }
    public int ImEndTokenId { get; set; }
    public HashSet<int> EosTokenIds { get; private set; } = [];
    public int PadTokenId { get; private set; }
    public bool HasThinkingMode { get; private set; }

    // ── Generation defaults ──────────────────────────────────────────────────
    public float DefaultTemperature { get; private set; } = 0.7f;
    public float DefaultTopP { get; private set; } = 0.9f;
    public int DefaultTopK { get; private set; } = 20;

    // ── Hardware (populated by HardwareDetector) ─────────────────────────────
    public int IntraOpThreads { get; set; } = 1;
    public int InterOpThreads { get; set; } = 1;

    // =========================================================================
    //  FACTORY
    // =========================================================================

    public static ModelConfig Load(string modelDir, ILogger logger)
    {
        var cfg = new ModelConfig { ModelDir = modelDir };
        cfg.ModelName = Path.GetFileName(modelDir.TrimEnd('/', '\\'));

        var configPath = Path.Combine(modelDir, "config.json");
        var genConfigPath = Path.Combine(modelDir, "generation_config.json");
        var tokConfigPath = Path.Combine(modelDir, "tokenizer_config.json");

        if (!File.Exists(configPath))
            throw new FileNotFoundException("config.json not found in model dir", configPath);

        // ── Parse config.json ────────────────────────────────────────────────
        var configDoc = JsonNode.Parse(File.ReadAllText(configPath))!;
        var modelType = configDoc["model_type"]?.GetValue<string>() ?? "";

        // Qwen3.5 wraps text_config inside a nested object
        JsonNode textCfg = configDoc["text_config"] ?? configDoc;

        cfg.NumHiddenLayers = textCfg["num_hidden_layers"]!.GetValue<int>();
        cfg.NumAttentionHeads = textCfg["num_attention_heads"]!.GetValue<int>();
        cfg.NumKeyValueHeads = textCfg["num_key_value_heads"]!.GetValue<int>();
        cfg.HeadDim = textCfg["head_dim"]?.GetValue<int>() ?? 128;
        cfg.HiddenSize = textCfg["hidden_size"]!.GetValue<int>();
        cfg.VocabSize = textCfg["vocab_size"]!.GetValue<int>();

        // ── Detect family ────────────────────────────────────────────────────
        cfg.Family = modelType.ToLowerInvariant() switch
        {
            "qwen3_5" => ModelFamily.Qwen3_5,
            var t when t.StartsWith("qwen3") => ModelFamily.Qwen3,
            _ => ModelFamily.Unknown
        };

        // ── Hybrid layer types (Qwen3.5) ─────────────────────────────────────
        if (textCfg["layer_types"] is JsonArray layerTypesArr)
        {
            var types = layerTypesArr.Select(x => x!.GetValue<string>()).ToList();
            cfg.LayerTypes = types;
            cfg.FullAttentionLayers = types
                .Select((t, i) => (t, i))
                .Where(x => x.t == "full_attention")
                .Select(x => x.i)
                .ToList();

            // mrope is used when rope_parameters.mrope_section is present
            cfg.UseMRope = textCfg["rope_parameters"]?["mrope_section"] != null;

            // Conv/recurrent dims read from config fields
            cfg.ConvKernelDim = textCfg["linear_conv_kernel_dim"]?.GetValue<int>() ?? 4;
            // ConvStateSize = linear_num_key_heads * hidden_size / num_attention_heads * ? 
            // Confirmed from tensor: [batch, 6144, 4]
            // 6144 = linear_num_value_heads(16) * linear_value_head_dim(128) * 3 = nope
            // Actually: 6144 = hidden_size(1024) * 6 nope
            // From tensor inspection: shape is always [batch, 6144, 4] — derive from model
            // We'll read actual shape from ONNX model inputs at runtime
            cfg.ConvStateSize = 0; // will be set by engine after session inspect
            cfg.RecurrentNumHeads = textCfg["linear_num_key_heads"]?.GetValue<int>() ?? 16;
            cfg.RecurrentKeyDim = textCfg["linear_key_head_dim"]?.GetValue<int>() ?? 128;
        }

        // ── ONNX file detection ──────────────────────────────────────────────
        cfg.DecoderModelFile = LocateDecoderModel(modelDir, cfg.Family);
        cfg.EmbedTokensFile = cfg.Family == ModelFamily.Qwen3_5
            ? LocateEmbedTokensModel(modelDir)
            : null;

        // ── generation_config.json ───────────────────────────────────────────
        if (File.Exists(genConfigPath))
        {
            var genDoc = JsonNode.Parse(File.ReadAllText(genConfigPath))!;
            cfg.DefaultTemperature = genDoc["temperature"]?.GetValue<float>() ?? 0.7f;
            cfg.DefaultTopP = genDoc["top_p"]?.GetValue<float>() ?? 0.9f;
            cfg.DefaultTopK = genDoc["top_k"]?.GetValue<int>() ?? 20;

            // EOS token ids — can be int or array
            var eosNode = genDoc["eos_token_id"];
            if (eosNode is JsonArray eosArr)
                foreach (var e in eosArr) cfg.EosTokenIds.Add(e!.GetValue<int>());
            else if (eosNode != null)
                cfg.EosTokenIds.Add(eosNode.GetValue<int>());

            cfg.PadTokenId = genDoc["pad_token_id"]?.GetValue<int>() ?? 0;
        }

        // ── tokenizer_config.json ────────────────────────────────────────────
        if (File.Exists(tokConfigPath))
        {
            var tokDoc = JsonNode.Parse(File.ReadAllText(tokConfigPath))!;
            // im_start/im_end discovered from tokenizer.json at runtime by tokenizer
            // just check for thinking mode
            var chatTemplate = tokDoc["chat_template"]?.GetValue<string>() ?? "";
            cfg.HasThinkingMode = chatTemplate.Contains("enable_thinking");
        }

        // Fallback EOS if none found
        if (cfg.EosTokenIds.Count == 0)
            cfg.EosTokenIds = [151645, 151643];

        logger.LogInformation("[ModelConfig] Family={F} Layers={L} KVHeads={KV} HeadDim={HD} Vocab={V}",
            cfg.Family, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim, cfg.VocabSize);
        logger.LogInformation("[ModelConfig] FullAttnLayers=[{FA}] MRope={MR} EOS=[{EOS}]",
            string.Join(",", cfg.FullAttentionLayers),
            cfg.UseMRope,
            string.Join(",", cfg.EosTokenIds));

        return cfg;
    }

    // =========================================================================
    //  HELPERS
    // =========================================================================

    private static string LocateDecoderModel(string dir, ModelFamily family)
    {
        // Priority: q4f16 (best) → q4 → quantized(int8) → fp16 → full
        var candidates = family == ModelFamily.Qwen3_5
            ? new[]
            {
                "decoder_model_merged_q4f16.onnx",
                "decoder_model_merged_q4.onnx",
                "decoder_model_merged_quantized.onnx",
                "decoder_model_merged_fp16.onnx",
                "decoder_model_merged.onnx"
            }
            : new[]
            {
                "model_q4f16.onnx",
                "model_q4.onnx",
                "model_int4.onnx",
                "model_quantized.onnx",
                "model.onnx"
            };

        foreach (var c in candidates)
        {
            var p = Path.Combine(dir, c);
            if (File.Exists(p)) return p;
        }

        // Last resort: any .onnx that's not embed_tokens or vision
        return Directory.GetFiles(dir, "*.onnx")
            .Where(f => !Path.GetFileName(f).Contains("embed_tokens") &&
                        !Path.GetFileName(f).Contains("vision"))
            .FirstOrDefault()
            ?? throw new FileNotFoundException("No decoder .onnx found in: " + dir);
    }

    private static string LocateEmbedTokensModel(string dir)
    {
        // Priority: q4f16 → q4 → quantized → fp16 → full
        var candidates = new[]
        {
            "embed_tokens_q4f16.onnx",
            "embed_tokens_q4.onnx",
            "embed_tokens_quantized.onnx",
            "embed_tokens_fp16.onnx",
            "embed_tokens.onnx"
        };

        foreach (var c in candidates)
        {
            var p = Path.Combine(dir, c);
            if (File.Exists(p)) return p;
        }
        throw new FileNotFoundException("No embed_tokens .onnx found in: " + dir);
    }
}