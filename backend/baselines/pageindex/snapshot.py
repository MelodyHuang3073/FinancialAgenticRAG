"""Safety copy of a multi-day PageIndex run. Run it after every session (or every day):

    python snapshot.py --run-dir <train_run> --backup-dir <backups> --out <experiment_results/baseline_pageindex> \
        --ledger-since "2026-11-15 22:56:00" [--commit]

1. copies results_*.json, index_*.json and the logs to <backup-dir>/<timestamp>/ (nothing is overwritten) and zips the index
   stores (the trees: re-indexing costs about 2e5 tokens per filing, so losing them is the expensive loss);
2. runs merge_results.py: answers_partial.json, index_summary.json, progress.json and an append-only copy of the usage ledger
   inside the repo folder, so the cost record survives a cleanup of backend/.llm_usage;
3. with --commit: commits that repo folder on the current branch and pushes it (a work-in-progress commit; they are squashed
   into the one results commit at the end, with the user's approval for the force-push).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(BACKEND)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger-since", required=True)
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    dest = os.path.join(args.backup_dir, time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for pat in ("results_*.json", "index_*.json", "run_*.log"):
        for f in glob.glob(os.path.join(args.run_dir, pat)):
            shutil.copy2(f, dest)
    for store in glob.glob(os.path.join(args.run_dir, "store_*")):
        shutil.make_archive(os.path.join(dest, os.path.basename(store)), "zip", store)
    print("backup ->", dest)
    subprocess.run([sys.executable, os.path.join(HERE, "merge_results.py"), "--run-dir", args.run_dir, "--out", args.out,
                    "--ledger-since", args.ledger_since], check=True)
    if args.commit:
        p = json.load(open(os.path.join(args.out, "progress.json"), encoding="utf-8"))
        subprocess.run(["git", "-C", REPO, "add", args.out], check=True)
        msg = f"PageIndex 訓練集進行中:已回答 {p['questions']} 題、已建索引 {p['filings_indexed']} 份財報、用量 {p['tokens_in_ledger']:,} token"
        if subprocess.run(["git", "-C", REPO, "commit", "-m", msg]).returncode == 0:
            subprocess.run(["git", "-C", REPO, "push"], check=True)
        else:
            print("nothing new to commit")


if __name__ == "__main__":
    main()
