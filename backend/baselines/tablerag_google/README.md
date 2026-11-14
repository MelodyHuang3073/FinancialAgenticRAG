# Baseline: Google TableRAG on our 300 questions

Paper: Chen et al., *TableRAG: Million-Token Table Understanding with Language Models*, NeurIPS 2024 (arXiv 2410.04739).
Code: `google-research/google-research`, folder `table_rag` (Apache-2.0 file headers). It is **not copied into this repository**:
download the folder (GitHub contents API at a fixed commit) and point `TABLERAG_SRC` at it.
The copy used so far is google-research master at `4b61cfbcd56b19e80ed5d95a80bac58f5ea70217`.

TableRAG only takes **one table per question and no text**. Our adaptation:

| File | Role |
|---|---|
| `build_tables.py` | one long table per filing from the rows our parser extracted (`page, line_item, period, value, number, unit`) + the 300 questions as JSONL |
| `compat.py` | makes Google's code run on gpt-5-mini without editing it (every patch is listed in its docstring) |
| `fin_prompts.py` | prompts for the financial task: the wtq prompts plus a table description and an answer-form note |
| `run_questions.py` | runs the questions (single-file, `--ids`, `--shard`, `--resume`), results in the other experiments' `answers.json` layout |

## Environment (separate from the backend venv)
`requirements.txt` here lists the tested versions. **pandas must stay 2.x**: with pandas 3 the text columns are no longer
`object`, Google's `build_cell_corpus` skips them and the cell database holds only min/max rows
(`test_cell_corpus_sees_text_columns` guards this).

```
python -m venv venv && venv/Scripts/python -m pip install -r backend/baselines/tablerag_google/requirements.txt
set TABLERAG_SRC=<path>\table_rag
cd backend
python -m baselines.tablerag_google.build_tables --out <data dir>
python -m baselines.tablerag_google.run_questions --data-dir <data dir> --out <results.json> --ids train-1,train-2
```
The tests (`tests/test_tablerag_google_*.py`) are offline and are skipped when these packages or the source are missing.

## Known limits (to report with the results)
- Text passages are not used, so questions whose evidence is narrative can only be answered from table numbers.
- 16 of the 300 questions sit on filings with fewer than 300 extracted table cells (our parser's limit, not the method's).
- The model is gpt-5-mini; `temperature`, `top_p`, `stop` and `max_tokens` are not accepted by it and are handled in `compat.py`.
- Embedding calls (`text-embedding-3-large`) are not in the usage ledger (the library returns no usage); the estimate for all 115 filings is about 3.0M tokens.
