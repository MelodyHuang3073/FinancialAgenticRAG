"""Local LLM token-usage ledger shared by EVERY place that calls a model.

OpenAI never says "this call was billed against the paid balance instead of the
free daily grant", so the account is tracked locally. Two call sites exist today:
the answer generator (llm_client.py) and the query decomposer (decomposer.py);
both report here through `record_usage`. Any new code that calls a model must
call it too.

The ledger is append-only JSON lines (one line per call), ONE FILE PER PROCESS
(.llm_usage/<date>_<pid>.jsonl), so parallel processes (the sharded regression
runs 6) never write to the same file: the previous read-modify-write daily JSON
lost updates, and even O_APPEND writes to one shared file dropped ~4% of the lines
when six processes wrote at once on Windows. The daily total sums every file of
the day.

Check it any time:  python -m app.agent.usage_tracker   (run from backend/)
"""
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USAGE_DIR = os.path.join(BASE_DIR, ".llm_usage")

#: Daily free allowance for mini/nano models (10,000,000 tokens/day, confirmed by
#: the user against their own OpenAI account).
DAILY_FREE_TOKEN_CAP = 10_000_000
WARN_FRACTION = 0.8


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _usage_numbers(usage: Any) -> Dict[str, Optional[int]]:
    """Token counts from an OpenAI `usage` object or a Gemini `usage_metadata`."""
    if usage is None:
        return {"prompt": None, "completion": None, "reasoning": None, "total": None}

    def g(*names):
        for n in names:
            v = getattr(usage, n, None)
            if v is None and isinstance(usage, dict):
                v = usage.get(n)
            if v is not None:
                return int(v)
        return None

    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
    return {
        "prompt": g("prompt_tokens", "prompt_token_count"),
        "completion": g("completion_tokens", "candidates_token_count"),
        "reasoning": int(reasoning) if reasoning is not None else None,
        "total": g("total_tokens", "total_token_count"),
    }


def _read_today() -> Dict[str, Any]:
    day = _today()
    total = calls = unknown = 0
    try:
        files = [
            os.path.join(USAGE_DIR, n) for n in os.listdir(USAGE_DIR)
            if n.startswith(day + "_") and n.endswith(".jsonl")
        ]
    except FileNotFoundError:
        files = []
    by_caller: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "tokens": 0})
    by_model: Dict[str, int] = defaultdict(int)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("date") != day:
                        continue
                    calls += 1
                    t = r.get("total_tokens")
                    if t is None:
                        unknown += 1  # call happened but the API returned no usage
                        t = 0
                    total += t
                    by_caller[r.get("caller", "?")]["calls"] += 1
                    by_caller[r.get("caller", "?")]["tokens"] += t
                    by_model[r.get("model", "?")] += t
        except OSError:
            continue
    return {"date": day, "total_tokens": total, "calls": calls, "calls_without_usage": unknown,
            "by_caller": dict(by_caller), "by_model": dict(by_model)}


def today_total() -> int:
    return _read_today()["total_tokens"]


def _notify_once(kind: str, message: str) -> None:
    """Print `message` to stderr the first time `kind` fires today (any process)."""
    marker = os.path.join(BASE_DIR, f".llm_usage_notified_{kind}_{_today()}")
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    except Exception:
        pass  # notifying is best-effort
    print(message, file=sys.stderr, flush=True)


def record_usage(model: str, caller: str, usage: Any = None, prompt_chars: int = 0) -> None:
    """Append one ledger line for a model call. `usage` is the API response's own
    usage object (may be None: the call is still counted, with unknown tokens)."""
    try:
        n = _usage_numbers(usage)
        line = json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "date": _today(),
            "pid": os.getpid(),
            "model": model,
            "caller": caller,
            "prompt_tokens": n["prompt"],
            "completion_tokens": n["completion"],
            "reasoning_tokens": n["reasoning"],
            "total_tokens": n["total"],
            "prompt_chars": prompt_chars,
        }, ensure_ascii=False) + "\n"
        os.makedirs(USAGE_DIR, exist_ok=True)
        with open(os.path.join(USAGE_DIR, f"{_today()}_{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
            f.write(line)

        total = today_total()
        if total >= DAILY_FREE_TOKEN_CAP:
            _notify_once(
                "over",
                f"[LLM DAILY FREE QUOTA] model='{model}': {total:,} tokens used today, past the "
                f"{DAILY_FREE_TOKEN_CAP:,}-token/day free-tier cap -- further calls are billed "
                f"against your paid balance.",
            )
        elif total >= WARN_FRACTION * DAILY_FREE_TOKEN_CAP:
            _notify_once(
                "warn",
                f"[LLM DAILY FREE QUOTA] {total:,} tokens used today "
                f"({total * 100 // DAILY_FREE_TOKEN_CAP}% of the {DAILY_FREE_TOKEN_CAP:,} free cap).",
            )
    except Exception:
        pass  # tracking is best-effort; never let it break a real LLM call


def format_report() -> str:
    d = _read_today()
    pct = d["total_tokens"] * 100 / DAILY_FREE_TOKEN_CAP
    out = [
        f"{d['date']}: {d['total_tokens']:,} tokens in {d['calls']} calls "
        f"({pct:.0f}% of the {DAILY_FREE_TOKEN_CAP:,} free cap)"
        + ("  ** OVER THE FREE CAP **" if d["total_tokens"] >= DAILY_FREE_TOKEN_CAP else ""),
    ]
    for caller, v in sorted(d["by_caller"].items()):
        out.append(f"  {caller:<12} {v['calls']:>6} calls  {v['tokens']:>12,} tokens")
    for model, t in sorted(d["by_model"].items()):
        out.append(f"  model {model}: {t:,}")
    if d["calls_without_usage"]:
        out.append(f"  {d['calls_without_usage']} call(s) returned no usage data (not in the total)")
    return "\n".join(out)


if __name__ == "__main__":
    print(format_report())
