"""
Runs the project's "information extraction" question set — a separate
regression group from run_calc_questions.py's "計算題" set.

FinanceBench's own dataset labels each question with a `question_reasoning`
field ("Information extraction" / "Numerical reasoning" / "Logical reasoning
(based on numerical reasoning)", or an "X OR Y" hybrid when the original
authors weren't sure which single label applied). That field is NOT present
in this project's local financebench_qa_subset.json (only doc_name/question/
answer/question_type are), so it was fetched once from the official dataset
(patronus-ai/financebench, data/financebench_open_source.jsonl) to build this
list — see EXTRACTION_QUESTIONS below for the result.

Of FinanceBench's 150 questions, 37 carry "Information extraction" as their
question_reasoning (31 purely, 6 as one option in an "X OR Y" hybrid label).
6 of those 37 are ALSO already covered by run_calc_questions.py's
CALC_QUESTIONS (FinanceBench tagged them "Information extraction" even
though they ask for one specific computed/extracted number that test already
tracks) and are deliberately NOT duplicated here. The remaining 31 — now all
with a matching PDF fixture in tests/financebench_pdfs/ — are
EXTRACTION_QUESTIONS below.

EXTRACTION_QUESTIONS is the single source of truth for which questions this
is — to add or remove one, edit the list directly. Gold answers and
question_type are looked up from financebench_qa_subset.json by exact
(doc_name, question) match at run time, so they never need to be retyped
here and always stay in sync with that file.

Usage:
    python tests/run_extraction_questions.py
Writes extraction_results.json (backend/extraction_results.json) with full
per-question detail; prints a pass/fail table to stdout.
"""
import sys, os, json, time

# Extraction gold answers routinely quote filing text verbatim (curly
# quotes, em dashes, non-ASCII currency/ticker symbols), which crashes a
# plain print() under Windows' default console codepage (cp950/cp1252 —
# neither is UTF-8) with UnicodeEncodeError, killing the whole run
# partway through and losing every remaining question's result. Forcing
# stdout/stderr to UTF-8 (falling back to replacing any still-unencodable
# byte rather than raising) makes every print() safe regardless of the
# console's own codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_financebench_qa import (
    QA_PATH, _build_indexed_store, _check_numeric, _check_contains_facts,
    _available_doc_names,
)
from app.agent.orchestrator import FinAgentRAGOrchestrator

# The information-extraction question set — (doc_name, exact question text).
# All are FinanceBench questions tagged question_reasoning="Information
# extraction" (or an "X OR Y" hybrid naming it): the answer is a fact,
# figure, or list read directly out of the filing (narrative sections like
# Item 1 Business/MD&A, or a single balance-sheet/income-statement line),
# not a formula computed by combining several line items.
EXTRACTION_QUESTIONS = [
    ("3M_2018_10K",
     "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a "
     "response to the question by relying on the details shown in the cash flow "
     "statement."),
    ("3M_2018_10K",
     "Assume that you are a public equities analyst. Answer the following question by "
     "primarily using information that is shown in the balance sheet: what is the year "
     "end FY2018 net PPNE for 3M? Answer in USD billions."),
    ("3M_2023Q2_10Q",
     "Which debt securities are registered to trade on a national securities exchange "
     "under 3M's name as of Q2 of 2023?"),
    ("AES_2022_10K",
     "What is the quantity of restructuring costs directly outlined in AES Corporation's "
     "income statements for FY2022? If restructuring costs are not explicitly outlined "
     "then state 0."),
    ("AMAZON_2019_10K",
     "By drawing conclusions from the information stated only in the income statement, "
     "what is Amazon's FY2019 net income attributable to shareholders (in USD "
     "millions)?"),
    ("AMCOR_2020_10K",
     "What is Amcor's year end FY2020 net AR (in USD millions)? Address the question by "
     "adopting the perspective of a financial analyst who can only use the details shown "
     "within the balance sheet."),
    ("AMCOR_2023_10K",
     "What are major acquisitions that AMCOR has done in FY2023, FY2022 and FY2021?"),
    ("AMCOR_2023_10K",
     "What industry does AMCOR primarily operate in?"),
    ("AMD_2022_10K",
     "What are the major products and services that AMD sells as of FY22?"),
    ("AMD_2022_10K",
     "What drove revenue change as of the FY22 for AMD?"),
    ("AMERICANEXPRESS_2022_10K",
     "Which debt securities are registered to trade on a national securities exchange "
     "under American Express' name as of 2022?"),
    ("AMERICANEXPRESS_2022_10K",
     "What are the geographies that American Express primarily operates in as of 2022?"),
    ("AMERICANEXPRESS_2022_10K",
     "Does AMEX have an improving operating margin profile as of 2022? If operating "
     "margin is not a useful metric for a company like this, then state that and "
     "explain why."),
    ("BESTBUY_2023_10K",
     "What are major acquisitions that Best Buy has done in FY2023, FY2022 and FY2021?"),
    ("BOEING_2022_10K",
     "Has Boeing reported any materially important ongoing legal battles from FY2022?"),
    ("BOEING_2022_10K",
     "Does Boeing have an improving gross margin profile as of FY2022? If gross margin "
     "is not a useful metric for a company like this, then state that and explain why."),
    ("BOEING_2022_10K",
     "Who are the primary customers of Boeing as of FY2022?"),
    ("COSTCO_2021_10K",
     "Using only the information within the balance sheet, how much total assets did "
     "Costco have at the end of FY2021? Answer in USD millions."),
    ("CVSHEALTH_2022_10K",
     "Has CVS Health reported any materially important ongoing legal battles from 2022, "
     "2021 and 2020?"),
    ("CVSHEALTH_2022_10K",
     "Has CVS Health paid dividends to common shareholders in Q2 of FY2022?"),
    ("MGMRESORTS_2018_10K",
     "Basing your judgments off of the balance sheet, what is the year end FY2018 amount "
     "of accounts payable for MGM Resorts? Answer in USD millions."),
    ("MGMRESORTS_2022_10K",
     "Has MGM Resorts paid dividends to common shareholders in FY2022?"),
    ("MICROSOFT_2016_10K",
     "What is the FY2016 COGS for Microsoft? Please state answer in USD millions. "
     "Provide a response to the question by primarily using the statement of income."),
    ("NETFLIX_2017_10K",
     "What is Netflix's year end FY2017 total current liabilities (in USD millions)? "
     "Base your judgments on the information provided primarily in the balance sheet."),
    ("NIKE_2019_10K",
     "According to the details clearly outlined within the balance sheet, how much "
     "total current assets did Nike have at the end of FY2019? Answer in USD millions."),
    ("PEPSICO_2021_10K",
     "What is the FY2021 capital expenditure amount (in USD billions) for PepsiCo? "
     "Respond to the question by assuming the perspective of an investment analyst who "
     "can only use the details shown within the statement of cash flows."),
    ("PEPSICO_2022_10K",
     "What are the geographies that Pepsico primarily operates in as of FY2022?"),
    ("PEPSICO_2022_10K",
     "Has Pepsico reported any materially important ongoing legal battles from FY2022 "
     "and FY2021?"),
    ("PEPSICO_2022_10K",
     "What is the quantity of restructuring costs directly outlined in Pepsico's income "
     "statements for FY2022? If restructuring costs are not explicitly outlined then "
     "state 0."),
    ("ULTABEAUTY_2023_10K",
     "Which debt securities are registered to trade on a national securities exchange "
     "under Ulta Beauty's name as of FY2023?"),
    ("ULTABEAUTY_2023_10K",
     "What are major acquisitions that Ulta Beauty has done in FY2023 and FY2022?"),
]


def load_extraction_qa_pairs():
    """Look up gold answer / question_type for each EXTRACTION_QUESTIONS
    entry from financebench_qa_subset.json by exact (doc_name, question)
    match."""
    with open(QA_PATH, "r", encoding="utf-8") as f:
        all_qa = {(qa["doc_name"], qa["question"]): qa for qa in json.load(f)}

    resolved = []
    for doc_name, question in EXTRACTION_QUESTIONS:
        qa = all_qa.get((doc_name, question))
        if qa is None:
            raise KeyError(
                f"EXTRACTION_QUESTIONS entry not found in {QA_PATH} (doc_name/question text "
                f"drifted out of sync): [{doc_name}] {question[:80]}"
            )
        resolved.append(qa)
    return resolved


def main():
    extraction_pairs = load_extraction_qa_pairs()
    available = _available_doc_names()
    runnable = [qa for qa in extraction_pairs if qa["doc_name"] in available]
    skipped = [qa for qa in extraction_pairs if qa["doc_name"] not in available]

    print(f"Extraction question set: {len(extraction_pairs)} total "
          f"({len(runnable)} runnable, {len(skipped)} skipped — no PDF fixture)")
    if skipped:
        for qa in skipped:
            print(f"  SKIP (no PDF): [{qa['doc_name']}] {qa['question'][:70]}")

    t0 = time.time()
    vs = _build_indexed_store()
    print(f"Indexing took {time.time() - t0:.1f}s, total passages: {len(vs.corpus)}")

    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    results = []
    for i, qa in enumerate(runnable, 1):
        q = qa["question"]
        gold = qa["answer"]
        doc = qa["doc_name"]
        qtype = qa.get("question_type")
        t1 = time.time()
        try:
            res = orchestrator.process_query(q)
            model_answer = res.get("final_answer", "") or ""
            error = None
        except Exception as e:
            model_answer = ""
            error = repr(e)
        elapsed = time.time() - t1

        if qtype == "metrics-generated":
            passed = _check_numeric(gold, model_answer)
            check_kind = "numeric (2% tol)"
        else:
            fact_check = _check_contains_facts(gold, model_answer)
            passed = fact_check if fact_check is not None else None
            check_kind = "fact-presence (2% tol)" if fact_check is not None else "informational only"

        results.append({
            "doc_name": doc, "question": q, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": check_kind, "error": error, "elapsed": elapsed,
        })
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"\n[{i}/{len(runnable)}] {status} ({elapsed:.1f}s) [{doc}]")
        print(f"  Q: {q[:100]}")
        print(f"  Gold : {gold}")
        print(f"  Model: {model_answer[:250]}")
        if error:
            print(f"  ERROR: {error}")

    scored = [r for r in results if r["passed"] is not None]
    n_pass = sum(1 for r in scored if r["passed"])
    print(f"\n{'=' * 70}")
    print(f"EXTRACTION QUESTIONS RESULT: {n_pass}/{len(scored)} passed"
          + (f" ({len(results) - len(scored)} informational-only, not scored)"
             if len(results) > len(scored) else ""))
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "extraction_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
