#!/usr/bin/env python3
"""
chat.py -- a terminal chat client for the OpenAI-compatible server in `server/app.py`.

Standard library only, streaming, multi-turn. After each reply it prints the engine's own
`x_engine_stats` for that request -- decode tok/s, the DSpark acceptance length it was achieved at,
and the expert hit rate -- because on this model a tok/s number without the acceptance beside it
says almost nothing: the same build measures 20 tok/s at acceptance 3.0 and 30 at 4.7.

Usage:
    python measure/chat.py                      # http://127.0.0.1:8100
    python measure/chat.py --url http://host:8100 --thinking

Commands inside the chat:
    /think on|off   toggle reasoning mode for the next turns
    /effort N       reasoning effort 1-100 (thinking mode only)
    /temp X         sampling temperature (0 = greedy)
    /system TEXT    replace the system prompt
    /clear          forget the conversation
    /stats          show the last reply's engine stats again
    /quit           leave
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BOLD, DIM, CYAN, YELLOW, GREY, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[90m", "\033[0m"


def post_stream(url: str, payload: dict, timeout: float = 3600.0):
    """Yield the parsed `data:` objects of an OpenAI-style SSE stream."""
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                continue


_PHASES = [
    ("weights: all non-expert weights on GPU", "non-expert weights on GPU"),
    ("DSpark experts resident", "DSpark drafter resident"),
    ("warm start done", "arena filled"),
    ("fast decode path enabled", "CUDA graphs built"),
]


def _progress_from_log(path: str) -> str | None:
    """What the engine is doing, read out of its own log.

    The server binds the port only after the expert arena is full, which for the all-resident plan
    is about four minutes of NVMe reads and GPU re-packing. A spinner cannot tell that apart from a
    server that is never coming up, and the engine is already printing exactly the right numbers.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    last = None
    for line in reversed(lines):
        if "Traceback" in line or "Error:" in line:
            return f"{YELLOW}the engine logged an error -- see {path}{RESET}"
        if "warm start " in line and "/" in line and "done" not in line:
            try:
                frag = line.split("warm start ", 1)[1]
                done, rest = frag.split("/", 1)
                total = rest.split(" ", 1)[0]
                gb = rest.split(",")[1].strip() if "," in rest else ""
                pct = 100.0 * int(done) / max(1, int(total))
                bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
                return f"filling the expert arena {bar} {pct:5.1f}%  {done}/{total} experts, {gb}"
            except (ValueError, IndexError):
                return None
        if "weights: layer" in line and last is None:
            try:
                n = int(line.split("weights: layer ", 1)[1].split(" ", 1)[0])
                last = f"loading the non-expert weights: layer {n}/39"
            except (ValueError, IndexError):
                pass
        for needle, human in _PHASES:
            if needle in line:
                return human
    return last


def wait_for_health(url: str, patience: float, log_path: str | None) -> dict | None:
    """The warm start fills the expert arena before the socket answers; several minutes is normal."""
    t0 = time.time()
    spin = "|/-\\"
    i = 0
    width = 0
    while time.time() - t0 < patience:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                print("\r" + " " * width + "\r", end="")
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            el = int(time.time() - t0)
            what = _progress_from_log(log_path) if log_path else None
            if what is None:
                what = "waiting for the engine (the expert arena fills before the port opens)"
            msg = f"{DIM}{spin[i % 4]} {el // 60}m{el % 60:02d}s  {what}{RESET}"
            width = max(width, len(msg))
            print("\r" + msg.ljust(width), end="", flush=True)
            i += 1
            time.sleep(2)
    return None


def fmt_stats(st: dict) -> str:
    if not st:
        return ""
    bits = []
    if st.get("decode_tok_s") is not None:
        bits.append(f"{st['decode_tok_s']:.2f} tok/s")
    if st.get("accept_len_mean") is not None:
        bits.append(f"acceptance {st['accept_len_mean']}")
    if st.get("completion_tokens") is not None:
        bits.append(f"{st['completion_tokens']} tok")
    if st.get("prefill_s") is not None:
        bits.append(f"TTFT {st['prefill_s']:.2f}s")
    if st.get("expert_hit_rate") is not None:
        bits.append(f"hit {st['expert_hit_rate']}")
    if st.get("nvme_gb"):
        bits.append(f"NVMe {st['nvme_gb']} GB")
    return " · ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--system", default="")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--effort", type=int, default=75)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--wait", type=float, default=1800.0, help="seconds to wait for /health")
    ap.add_argument("--log", default="logs/server.log",
                    help="the server's log, read to show real load progress while waiting; "
                         "pass '' to disable")
    a = ap.parse_args()

    health = wait_for_health(a.url, a.wait, a.log or None)
    if health is None:
        print(f"{YELLOW}the server at {a.url} never answered /health.{RESET}")
        print("Check the log:  tail -f logs/server.log")
        return 1

    cfg = health.get("engine_config", health) or {}
    slots = cfg.get("arena_slots")
    resident = f"{slots} slots" if slots else "?"
    print(f"{BOLD}DeepSeek-V4.1-Flash{RESET} on one DGX Spark")
    print(f"{DIM}{a.url} · {cfg.get('kernel', '?')} · arena {cfg.get('arena_gb', '?')} GB = {resident}"
          f"{' · every routed expert resident' if slots and slots >= 15360 else ''}"
          f" · max_seq {cfg.get('max_seq', '?')}{RESET}")
    print(f"{DIM}/think on|off  /effort N  /temp X  /system TEXT  /clear  /stats  /quit{RESET}\n")

    msgs: list[dict] = []
    if a.system:
        msgs.append({"role": "system", "content": a.system})
    thinking, effort, temp = a.thinking, a.effort, a.temp
    last_stats: dict = {}

    while True:
        try:
            line = input(f"{BOLD}{CYAN}you{RESET} ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("quit", "exit", "q"):
                return 0
            if cmd == "clear":
                msgs = [m for m in msgs if m["role"] == "system"]
                print(f"{DIM}conversation cleared{RESET}\n"); continue
            if cmd == "think":
                thinking = arg.lower() in ("on", "1", "true", "yes")
                print(f"{DIM}thinking {'on' if thinking else 'off'}{RESET}\n"); continue
            if cmd == "effort":
                try: effort = max(1, min(100, int(arg)))
                except ValueError: print(f"{YELLOW}/effort takes 1-100{RESET}"); continue
                print(f"{DIM}effort {effort}{RESET}\n"); continue
            if cmd == "temp":
                try: temp = float(arg)
                except ValueError: print(f"{YELLOW}/temp takes a number{RESET}"); continue
                print(f"{DIM}temperature {temp}{RESET}\n"); continue
            if cmd == "system":
                msgs = [m for m in msgs if m["role"] != "system"]
                if arg: msgs.insert(0, {"role": "system", "content": arg})
                print(f"{DIM}system prompt {'set' if arg else 'cleared'}{RESET}\n"); continue
            if cmd == "stats":
                print(f"{DIM}{json.dumps(last_stats, indent=1)}{RESET}\n"); continue
            print(f"{YELLOW}unknown command /{cmd}{RESET}\n"); continue

        msgs.append({"role": "user", "content": line})
        payload = {
            "model": a.model, "messages": msgs, "stream": True,
            "max_tokens": a.max_tokens, "temperature": temp,
        }
        if thinking:
            payload["chat_template_kwargs"] = {"thinking": True}
            payload["reasoning_effort"] = effort

        print(f"{BOLD}{YELLOW}model{RESET} ❯ ", end="", flush=True)
        answer, in_reasoning, t0, first = [], False, time.time(), None
        try:
            for chunk in post_stream(a.url, payload):
                if chunk.get("x_engine_stats"):
                    last_stats = chunk["x_engine_stats"]
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    rc = d.get("reasoning_content")
                    if rc:
                        if not in_reasoning:
                            print(f"\n{GREY}[thinking] ", end="", flush=True); in_reasoning = True
                        print(rc, end="", flush=True)
                    c = d.get("content")
                    if c:
                        if in_reasoning:
                            print(f"{RESET}\n{BOLD}{YELLOW}model{RESET} ❯ ", end="", flush=True)
                            in_reasoning = False
                        if first is None:
                            first = time.time()
                        print(c, end="", flush=True)
                        answer.append(c)
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}")
        except urllib.error.HTTPError as e:
            print(f"\n{YELLOW}HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}{RESET}\n")
            msgs.pop(); continue
        except Exception as e:  # noqa: BLE001
            print(f"\n{YELLOW}{type(e).__name__}: {e}{RESET}\n")
            msgs.pop(); continue

        text = "".join(answer)
        msgs.append({"role": "assistant", "content": text})
        wall = time.time() - t0
        line_stats = fmt_stats(last_stats) or (f"{wall:.1f}s wall" if wall else "")
        if first is not None and "TTFT" not in line_stats:
            line_stats += f" · first token {first - t0:.2f}s"
        print(f"\n{DIM}{line_stats}{RESET}\n")


if __name__ == "__main__":
    sys.exit(main())
