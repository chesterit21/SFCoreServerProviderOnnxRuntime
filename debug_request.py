#!/usr/bin/env python3
"""
Debug script — kirim request kecil dulu, lihat exact response
"""
import http.client
import json
import socket

VPS_HOST = "[IP_ADDRESS]"
VPS_PORT = 5005
API_KEY  = "YOUR-API_KEY"

def test(label, content, stream=True, max_tokens=30):
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"  Content length: {len(content)} chars")
    print(f"{'='*60}")

    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "temperature": 0.3
    }).encode("utf-8")

    print(f"  Payload: {len(body)} bytes")

    try:
        conn = http.client.HTTPConnection(VPS_HOST, VPS_PORT, timeout=12000)
        conn.connect()

        # TCP keepalive
        sock = conn.sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type":  "application/json",
            "Accept":        "text/event-stream" if stream else "application/json",
            "Connection":    "keep-alive",
        })

        resp = conn.getresponse()
        print(f"  HTTP Status: {resp.status} {resp.reason}")
        print(f"  Headers:")
        for h, v in resp.getheaders():
            print(f"    {h}: {v}")

        # Read first 2KB of response
        data = resp.read(2048)
        print(f"\n  First 2KB response:")
        print(f"  {data.decode('utf-8', errors='replace')[:500]}")
        conn.close()
        return resp.status

    except ConnectionResetError as e:
        print(f"  ❌ ConnectionResetError: {e}")
        return None
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        return None

# ── Test 1: tiny request ─────────────────────────────────────
test("Tiny (50 chars)", "apa warna langit?", stream=True)

# ── Test 2: medium request (~1K tokens) ──────────────────────
test("Medium (1K tokens)", "hello " * 500 + " apa warna langit?", stream=True)

# ── Test 3: larger (~5K tokens) ──────────────────────────────
test("Large (4.5K tokens)", "hello world test " * 1500 + " apa warna langit?", stream=True)

# ── Test 4: larger (~5K tokens) ──────────────────────────────
test("Large (7.5K tokens)", "hello world test " * 2500 + ". Ayo apa warna langit?", stream=True)

# ── Test 5: larger (~9K tokens) ──────────────────────────────
test("Large (9K tokens)", "hello world test " * 3000 + ". Ayo apa warna langit?", stream=True)

# ── Test 6: larger (~10K tokens) ──────────────────────────────
test("Large (10K tokens)", "hello world test " * 3500 + ". Ayo apa warna langit?", stream=True)


# ── Test 7: larger (~26K tokens) ──────────────────────────────
test("Large (22K-26K tokens)", "Langit selalu cerah " * 6500 + ". Ayo apa warna langit? dan berapa banyak kata-kata Langit selalu cerah di ulangi?", stream=True)

# ── Test 8: larger (~32K tokens) ──────────────────────────────
test("Large (30K-35K tokens)", "Langit selalu cerah " * 8000 + ". Ayo apa warna langit? dan berapa banyak kata-kata Langit selalu cerah di ulangi?", stream=True)

# ── Test 9: larger (~51K tokens) ──────────────────────────────
test("Large (45K-50K tokens)", "Langit selalu cerah " * 12800 + ". Ayo apa warna langit? dan berapa banyak kata-kata Langit selalu cerah di ulangi?", stream=True)

# ── Test 10: larger (~64K tokens) ─────────────────────────────
ctx = "Rust adalah bahasa pemrograman sistem. " * 10000
test("Stage FINAL size (60K-64K est)", ctx + " Ringkas semua itu.", stream=True)

print("\nDone.")
