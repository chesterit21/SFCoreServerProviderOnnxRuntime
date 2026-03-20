using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Text.Unicode;

/// <summary>
/// Pure C# BPE Tokenizer — reads tokenizer.json directly, zero external dependencies.
/// Implements: NFC normalize → Regex pre-tokenize → ByteLevel → BPE merge → vocab lookup
/// Supports Qwen3 and Qwen3.5 tokenizer formats.
/// </summary>
public class QwenTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<(string, string), int> _mergeRanks;
    private readonly HashSet<string> _specialTokens;
    private readonly Dictionary<string, int> _specialTokenIds;

    // GPT-2 style byte → unicode char mapping
    private static readonly char[] ByteToChar = BuildByteToChar();
    private static readonly Dictionary<char, byte> CharToByte;

    static QwenTokenizer()
    {
        CharToByte = new Dictionary<char, byte>();
        for (int i = 0; i < 256; i++)
            CharToByte[ByteToChar[i]] = (byte)i;
    }

    private static char[] BuildByteToChar()
    {
        var result = new char[256];
        var direct = Enumerable.Range('!', '~' - '!' + 1)
            .Concat(Enumerable.Range(0xA1, 0xAC - 0xA1 + 1))
            .Concat(Enumerable.Range(0xAE, 0xFF - 0xAE + 1))
            .ToHashSet();

        int extra = 256;
        for (int b = 0; b < 256; b++)
        {
            if (direct.Contains(b))
                result[b] = (char)b;
            else
                result[b] = (char)(extra++);
        }
        return result;
    }

    // Qwen3/3.5 GPT-4 style pre-tokenizer regex
    private static readonly Regex PreTokenizeRegex = new Regex(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture
    );

    public int VocabSize => _vocab.Count;

    public QwenTokenizer(string tokenizerJsonPath)
    {
        _vocab = new Dictionary<string, int>(350000);
        _idToToken = new Dictionary<int, string>(350000);
        _mergeRanks = new Dictionary<(string, string), int>(350000);
        _specialTokens = new HashSet<string>();
        _specialTokenIds = new Dictionary<string, int>();

        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream, new JsonDocumentOptions
        {
            AllowTrailingCommas = true,
            CommentHandling = JsonCommentHandling.Skip
        });
        var root = doc.RootElement;

        // 1. Load vocab
        var vocabEl = root.GetProperty("model").GetProperty("vocab");
        foreach (var kv in vocabEl.EnumerateObject())
        {
            var id = kv.Value.GetInt32();
            _vocab[kv.Name] = id;
            _idToToken[id] = kv.Name;
        }

        // 2. Load added_tokens (special tokens)
        if (root.TryGetProperty("added_tokens", out var addedTokensEl))
        {
            foreach (var token in addedTokensEl.EnumerateArray())
            {
                var content = token.GetProperty("content").GetString()!;
                var id = token.GetProperty("id").GetInt32();
                _vocab[content] = id;
                _idToToken[id] = content;
                if (token.TryGetProperty("special", out var sp) && sp.GetBoolean())
                {
                    _specialTokens.Add(content);
                    _specialTokenIds[content] = id;
                }
            }
        }

        // 3. Load merges
        var mergesEl = root.GetProperty("model").GetProperty("merges");
        int rank = 0;
        foreach (var merge in mergesEl.EnumerateArray())
        {
            string a, b;
            if (merge.ValueKind == JsonValueKind.Array)
            {
                var arr = merge.EnumerateArray().ToArray();
                if (arr.Length < 2) { rank++; continue; }
                a = arr[0].GetString()!;
                b = arr[1].GetString()!;
            }
            else
            {
                var parts = merge.GetString()!.Split(' ', 2);
                if (parts.Length < 2) { rank++; continue; }
                a = parts[0];
                b = parts[1];
            }
            _mergeRanks[(a, b)] = rank++;
        }
    }

    // =========================================================================
    //  PUBLIC API
    // =========================================================================

    public int[] Encode(string text, bool addSpecialTokens = false)
    {
        if (string.IsNullOrEmpty(text)) return [];

        text = text.Normalize(NormalizationForm.FormC);
        var result = new List<int>(text.Length * 2);

        if (addSpecialTokens)
            result.Add(_specialTokenIds.GetValueOrDefault("<|im_start|>"));

        var chunks = SplitOnSpecialTokens(text);
        foreach (var (chunk, isSpecial) in chunks)
        {
            if (isSpecial)
            {
                if (_vocab.TryGetValue(chunk, out var sid))
                    result.Add(sid);
                continue;
            }

            var matches = PreTokenizeRegex.Matches(chunk);
            foreach (Match match in matches)
                result.AddRange(BpeEncode(match.Value));
        }

        if (addSpecialTokens)
            result.Add(_specialTokenIds.GetValueOrDefault("<|im_end|>"));

        return result.ToArray();
    }

    public string Decode(int[] ids)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (!_idToToken.TryGetValue(id, out var tok)) continue;
            if (_specialTokens.Contains(tok)) continue;
            sb.Append(tok);
        }
        return ByteLevelDecode(sb.ToString());
    }

    public string DecodeSingle(int id)
    {
        if (!_idToToken.TryGetValue(id, out var token)) return "";
        if (_specialTokens.Contains(token)) return "";
        return ByteLevelDecodeStrict(token);
    }

    public string GetTokenString(int id)
    {
        if (!_idToToken.TryGetValue(id, out var token)) return "";
        if (_specialTokens.Contains(token)) return "";
        return token;
    }

    /// <summary>
    /// Lookup a special token ID by its string (e.g. "&lt;|im_start|&gt;").
    /// Returns null if not found.
    /// </summary>
    public int? GetSpecialTokenId(string tokenStr)
        => _specialTokenIds.TryGetValue(tokenStr, out var id) ? id : null;

    public StreamingDecoder CreateStreamingDecoder() => new StreamingDecoder(CharToByte);

    public string DecodeIds(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (!_idToToken.TryGetValue(id, out var tok)) continue;
            if (_specialTokens.Contains(tok)) continue;
            sb.Append(tok);
        }
        return ByteLevelDecode(sb.ToString());
    }

    // =========================================================================
    //  PRIVATE HELPERS
    // =========================================================================

    private int[] BpeEncode(string word)
    {
        var bytes = Encoding.UTF8.GetBytes(word);
        var chars = bytes.Select(b => ByteToChar[b].ToString()).ToList();

        while (chars.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIdx = -1;
            for (int i = 0; i < chars.Count - 1; i++)
            {
                if (_mergeRanks.TryGetValue((chars[i], chars[i + 1]), out var r) && r < bestRank)
                { bestRank = r; bestIdx = i; }
            }
            if (bestIdx == -1) break;
            chars[bestIdx] = chars[bestIdx] + chars[bestIdx + 1];
            chars.RemoveAt(bestIdx + 1);
        }

        return chars.Select(t => _vocab.TryGetValue(t, out var id) ? id : 0).ToArray();
    }

    private string ByteLevelDecode(string encoded)
    {
        try
        {
            var bytes = new List<byte>(encoded.Length);
            foreach (char c in encoded)
                if (CharToByte.TryGetValue(c, out var b)) bytes.Add(b);
            return Encoding.UTF8.GetString(bytes.ToArray());
        }
        catch { return string.Empty; }
    }

    private string ByteLevelDecodeStrict(string encoded)
    {
        try
        {
            var bytes = new List<byte>(encoded.Length);
            foreach (char c in encoded)
                if (CharToByte.TryGetValue(c, out var b)) bytes.Add(b);
            if (bytes.Count == 0) return string.Empty;

            var decoder = Encoding.UTF8.GetDecoder();
            var byteArr = bytes.ToArray();
            int charCount = decoder.GetCharCount(byteArr, 0, byteArr.Length, flush: false);
            if (charCount == 0) return string.Empty;

            var chars = new char[charCount];
            decoder.GetChars(byteArr, 0, byteArr.Length, chars, 0, flush: false);
            return new string(chars);
        }
        catch { return string.Empty; }
    }

    private List<(string text, bool isSpecial)> SplitOnSpecialTokens(string text)
    {
        if (_specialTokens.Count == 0)
            return [(text, false)];

        var result = new List<(string, bool)>();
        var pattern = string.Join("|", _specialTokens.Select(Regex.Escape));
        var regex = new Regex(pattern);
        int lastIdx = 0;

        foreach (Match match in regex.Matches(text))
        {
            if (match.Index > lastIdx)
                result.Add((text[lastIdx..match.Index], false));
            result.Add((match.Value, true));
            lastIdx = match.Index + match.Length;
        }

        if (lastIdx < text.Length)
            result.Add((text[lastIdx..], false));

        return result;
    }
}

// ============================================================
//  StreamingDecoder — stateful multi-byte UTF-8 decoder
// ============================================================

public sealed class StreamingDecoder
{
    private readonly Dictionary<char, byte> _charToByte;
    private readonly System.Text.Decoder _utf8Decoder;
    private readonly byte[] _byteBuffer = new byte[16];
    private int _bufLen = 0;

    internal StreamingDecoder(Dictionary<char, byte> charToByte)
    {
        _charToByte = charToByte;
        _utf8Decoder = new UTF8Encoding(false, false).GetDecoder();
    }

    public string Decode(string tokenBpeStr)
    {
        if (string.IsNullOrEmpty(tokenBpeStr)) return "";

        foreach (char c in tokenBpeStr)
            if (_charToByte.TryGetValue(c, out var b) && _bufLen < _byteBuffer.Length)
                _byteBuffer[_bufLen++] = b;

        if (_bufLen == 0) return "";

        var charBuf = new char[_bufLen * 2];
        int charsWritten;
        try { charsWritten = _utf8Decoder.GetChars(_byteBuffer, 0, _bufLen, charBuf, 0, flush: false); }
        catch { _bufLen = 0; _utf8Decoder.Reset(); return ""; }

        if (charsWritten == 0) return "";

        _bufLen = 0;
        var result = new string(charBuf, 0, charsWritten);
        return result.Contains('\uFFFD') ? "" : result;
    }

    public string Flush()
    {
        if (_bufLen == 0) return "";
        var charBuf = new char[_bufLen * 2];
        int n;
        try { n = _utf8Decoder.GetChars(_byteBuffer, 0, _bufLen, charBuf, 0, flush: true); }
        catch { n = 0; }
        _bufLen = 0;
        _utf8Decoder.Reset();
        var result = new string(charBuf, 0, n);
        return result.Contains('\uFFFD') ? "" : result;
    }

    public void Reset() { _bufLen = 0; _utf8Decoder.Reset(); }
}