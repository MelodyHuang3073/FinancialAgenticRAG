"""Remove the failed records (error, empty, fallback-shaped answer) from the shard files of a run, so that
launching the run again re-runs exactly those questions. Nothing else is touched; the answers of the
questions that did run stay as they are, and the removed records are kept in <shard>.failed.json.

    python backend/llm_runner/drop_failed.py --run-dir <RunDir>          # lists, changes nothing
    python backend/llm_runner/drop_failed.py --run-dir <RunDir> --apply  # removes them from the shard files

A daily-quota failure makes the system answer with a fallback-shaped text; that is not a model answer and
must never be graded (plan section 8: only the failed questions are re-run after the quota is back).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from failed import failure  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    total = 0
    for tag in ("train_0", "train_1", "out_0", "out_1"):
        path = os.path.join(a.run_dir, tag + ".json")
        if not os.path.exists(path):
            continue
        rows = json.load(open(path, encoding="utf-8"))
        bad = [(r, failure(r)) for r in rows]
        bad = [(r, why) for r, why in bad if why]
        for r, why in bad:
            print(f"{tag}: {r['doc_name']} / {r['question'][:60]}: {why}")
        total += len(bad)
        if a.apply and bad:
            json.dump([r for r, _ in bad], open(os.path.join(a.run_dir, tag + ".failed.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            keep = [r for r in rows if not failure(r)]
            json.dump(keep, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{total} failed records" + (" removed from the shard files" if a.apply else " (use --apply to remove them)"))
