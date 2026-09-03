"""
Runs the project's standing "計算題" (calculation questions) regression set:
a fixed, explicit list of 34 (doc_name, question) pairs — see CALC_QUESTIONS
below — that the user has asked to track as one group whenever they say
"測試計算題" (test the calculation questions). Most are FinanceBench's own
"metrics-generated" questions (pure ratio/formula calculations); a handful
are "domain-relevant" questions the user explicitly folded in because each
still asks for one specific computed ratio (e.g. "Roughly how many times
has AES sold its inventory in FY2022?"), even though FinanceBench itself
tags them differently.

CALC_QUESTIONS is the single source of truth for which 34 questions this
is — to add or remove one, edit the list directly. Gold answers and
question_type are looked up from financebench_qa_subset.json by exact
(doc_name, question) match at run time, so they never need to be retyped
here and always stay in sync with that file.

Usage:
    python tests/run_calc_questions.py
Writes calc_results.json (backend/calc_results.json) with full per-question
detail; prints a pass/fail table to stdout.
"""
import sys, os, json, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from test_financebench_qa import (
    QA_PATH, _build_indexed_store, _check_numeric, _check_contains_facts,
    _available_doc_names,
)
from app.agent.orchestrator import FinAgentRAGOrchestrator

# The 34-question standing calc set — (doc_name, exact question text).
# 28 FinanceBench "metrics-generated" + 6 "domain-relevant" ratio questions
# the user asked to fold in (marked below).
CALC_QUESTIONS = [
    ("ACTIVISIONBLIZZARD_2019_10K",
     "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset "
     "turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and "
     "FY2019). Round your answer to two decimal places. Base your judgments on the "
     "information provided primarily in the statement of income and the statement of "
     "financial position."),
    ("ACTIVISIONBLIZZARD_2019_10K",
     "What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision "
     "Blizzard? Answer in units of percents and round to one decimal place. Calculate (or "
     "extract) the answer from the statement of income and the cash flow statement."),
    ("ADOBE_2015_10K",
     "You are an investment banker and your only resource(s) to answer the following "
     "question is (are): the statement of financial position and the cash flow statement. "
     "Here's the question: what is the FY2015 operating cash flow ratio for Adobe? "
     "Operating cash flow ratio is defined as: cash from operations / total current "
     "liabilities. Round your answer to two decimal places."),
    ("ADOBE_2016_10K",
     "What is Adobe's year-over-year change in unadjusted operating income from FY2015 to "
     "FY2016 (in units of percents and round to one decimal place)? Give a solution to the "
     "question by using the income statement."),
    ("ADOBE_2017_10K",
     "What is the FY2017 operating cash flow ratio for Adobe? Operating cash flow ratio is "
     "defined as: cash from operations / total current liabilities. Round your answer to "
     "two decimal places. Please utilize information provided primarily within the balance "
     "sheet and the cash flow statement."),
    # ── domain-relevant (folded in — asks for one specific ratio) ──
    ("ADOBE_2022_10K",
     "Does Adobe have an improving operating margin profile as of FY2022? If operating "
     "margin is not a useful metric for a company like this, then state that and explain "
     "why."),
    ("AES_2022_10K",
     "Roughly how many times has AES Corporation sold its inventory in FY2022? Calculate "
     "inventory turnover ratio for the FY2022; if conventional inventory management is not "
     "meaningful for the company then state that and explain why."),
    ("AES_2022_10K",
     "Based on the information provided primarily in the statement of financial position "
     "and the statement of income, what is AES's FY2022 return on assets (ROA)? ROA is "
     "defined as: FY2022 net income / (average total assets between FY2021 and FY2022). "
     "Round your answer to two decimal places."),
    ("AMAZON_2017_10K",
     "What is Amazon's FY2017 days payable outstanding (DPO)? DPO is defined as: 365 * "
     "(average accounts payable between FY2016 and FY2017) / (FY2017 COGS + change in "
     "inventory between FY2016 and FY2017). Round your answer to two decimal places. "
     "Address the question by using the line items and information shown within the "
     "balance sheet and the P&L statement."),
    ("AMAZON_2017_10K",
     "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of "
     "percents and round to one decimal place)? Calculate what was asked by utilizing the "
     "line items clearly shown in the statement of income."),
    # ── domain-relevant (folded in — asks for one specific ratio) ──
    ("AMCOR_2023_10K",
     "Has AMCOR's quick ratio improved or declined between FY2023 and FY2022? If the quick "
     "ratio is not something that a financial analyst would ask about a company like this, "
     "then state that and explain why."),
    ("AMCOR_2023_10K",
     "Does AMCOR have an improving gross margin profile as of FY2023? If gross margin is "
     "not a useful metric for a company like this, then state that and explain why."),
    ("AMD_2015_10K",
     "Answer the following question as if you are an equity research analyst and have lost "
     "internet connection so you do not have access to financial metric providers. "
     "According to the details clearly outlined within the P&L statement and the statement "
     "of cash flows, what is the FY2015 depreciation and amortization (D&A from cash flow "
     "statement) % margin for AMD?"),
    ("AMERICANWATERWORKS_2020_10K",
     "How much (in USD billions) did American Water Works pay out in cash dividends for "
     "FY2020? Compute or extract the answer by primarily using the details outlined in the "
     "statement of cash flows."),
    ("AMERICANWATERWORKS_2021_10K",
     "Basing your judgments off of the cash flow statement and the income statement, what "
     "is American Water Works's FY2021 unadjusted operating income + depreciation and "
     "amortization from the cash flow statement (unadjusted EBITDA) in USD millions?"),
    # ── domain-relevant (folded in — asks for one specific ratio) ──
    ("AMERICANWATERWORKS_2022_10K",
     "Does American Water Works have positive working capital based on FY2022 data? If "
     "working capital is not a useful or relevant metric for this company, then please "
     "state that and explain why."),
    ("BESTBUY_2017_10K",
     "In agreement with the information outlined in the income statement, what is the "
     "FY2015 - FY2017 3 year average net profit margin (as a %) for Best Buy? Answer in "
     "units of percents and round to one decimal place."),
    ("BESTBUY_2019_10K",
     "What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD "
     "millions. Base your judgments on the information provided primarily in the balance "
     "sheet."),
    ("BLOCK_2016_10K",
     "Considering the data in the balance sheet, what is Block's (formerly known as "
     "Square) FY2016 working capital ratio? Define working capital ratio as total current "
     "assets divided by total current liabilities. Round your answer to two decimal "
     "places."),
    ("BLOCK_2020_10K",
     "What is the FY2019 - FY2020 total revenue growth rate for Block (formerly known as "
     "Square)? Answer in units of percents and round to one decimal place. Approach the "
     "question asked by assuming the standpoint of an investment banking analyst who only "
     "has access to the statement of income."),
    ("BLOCK_2020_10K",
     "Using the cash flow statement, answer the following question to the best of your "
     "abilities: how much did Block (formerly known as Square) generate in cash flow from "
     "operating activities in FY2020? Answer in USD millions."),
    ("BOEING_2018_10K",
     "We need to calculate a financial metric by using information only provided within "
     "the balance sheet. Please answer the following question: what is Boeing's year end "
     "FY2018 net property, plant, and equipment (in USD millions)?"),
    ("COCACOLA_2017_10K",
     "What is the FY2017 return on assets (ROA) for Coca Cola? ROA is defined as: FY2017 "
     "net income / (average total assets between FY2016 and FY2017). Round your answer to "
     "two decimal places. Give a response to the question by relying on the details shown "
     "in the balance sheet and the P&L statement."),
    ("COCACOLA_2021_10K",
     "What is Coca Cola's FY2021 COGS % margin? Calculate what was asked by utilizing the "
     "line items clearly shown in the income statement."),
    ("COCACOLA_2022_10K",
     "What is Coca Cola's FY2022 dividend payout ratio (using total cash dividends paid "
     "and net income attributable to shareholders)? Round answer to two decimal places. "
     "Answer the question asked by assuming you only have access to information clearly "
     "displayed in the cash flow statement and the income statement."),
    ("CORNING_2020_10K",
     "Based on the information provided primarily in the balance sheet and the statement "
     "of income, what is FY2020 days payable outstanding (DPO) for Corning? DPO is defined "
     "as: 365 * (average accounts payable between FY2019 and FY2020) / (FY2020 COGS + "
     "change in inventory between FY2019 and FY2020). Round your answer to two decimal "
     "places."),
    ("CORNING_2021_10K",
     "Taking into account the information outlined in the income statement, what is the "
     "FY2019 - FY2021 3 year average unadjusted operating income % margin for Corning? "
     "Answer in units of percents and round to one decimal place."),
    # ── domain-relevant (folded in — asks for one specific ratio) ──
    ("CORNING_2022_10K",
     "How much has the effective tax rate of Corning changed between FY2021 and FY2022?"),
    ("CVSHEALTH_2018_10K",
     "What is the FY2018 fixed asset turnover ratio for CVS Health? Fixed asset turnover "
     "ratio is defined as: FY2018 revenue / (average PP&E between FY2017 and FY2018). "
     "Round your answer to two decimal places. Calculate what was asked by utilizing the "
     "line items clearly shown in the P&L statement and the balance sheet."),
    ("GENERALMILLS_2019_10K",
     "What is the FY2019 cash conversion cycle (CCC) for General Mills? CCC is defined as: "
     "DIO + DSO - DPO. DIO is defined as: 365 * (average inventory between FY2018 and "
     "FY2019) / (FY2019 COGS). DSO is defined as: 365 * (average accounts receivable "
     "between FY2018 and FY2019) / (FY2019 Revenue). DPO is defined as: 365 * (average "
     "accounts payable between FY2018 and FY2019) / (FY2019 COGS + change in inventory "
     "between FY2018 and FY2019). Round your answer to two decimal places. Address the "
     "question by using the line items and information shown within the income statement "
     "and the balance sheet."),
    ("GENERALMILLS_2020_10K",
     "By drawing conclusions from the information stated only in the statement of "
     "financial position, what is General Mills's FY2020 working capital ratio? Define "
     "working capital ratio as total current assets divided by total current liabilities. "
     "Round your answer to two decimal places."),
    ("GENERALMILLS_2020_10K",
     "According to the information provided in the statement of cash flows, what is the "
     "FY2020 free cash flow (FCF) for General Mills? FCF here is defined as: (cash from "
     "operations - capex). Answer in USD millions."),
    ("GENERALMILLS_2022_10K",
     "We want to calculate a financial metric. Please help us compute it by basing your "
     "answers off of the cash flow statement and the income statement. Here's the "
     "question: what is the FY2022 retention ratio (using total cash dividends paid and "
     "net income attributable to shareholders) for General Mills? Round answer to two "
     "decimal places."),
    # ── domain-relevant (folded in — asks for one specific ratio) ──
    ("JOHNSON_JOHNSON_2022_10K",
     "Roughly how many times has JnJ sold its inventory in FY2022? Calculate inventory "
     "turnover ratio for FY2022; if conventional inventory management is not meaningful "
     "for the company then state that and explain why."),
    ("KRAFTHEINZ_2019_10K",
     "What is Kraft Heinz's FY2019 inventory turnover ratio? Inventory turnover ratio is "
     "defined as: (FY2019 COGS) / (average inventory between FY2018 and FY2019). Round "
     "your answer to two decimal places. Please base your judgments on the information "
     "provided primarily in the balance sheet and the P&L statement."),
]


def load_calc_qa_pairs():
    """Look up gold answer / question_type for each CALC_QUESTIONS entry
    from financebench_qa_subset.json by exact (doc_name, question) match."""
    with open(QA_PATH, "r", encoding="utf-8") as f:
        all_qa = {(qa["doc_name"], qa["question"]): qa for qa in json.load(f)}

    resolved = []
    for doc_name, question in CALC_QUESTIONS:
        qa = all_qa.get((doc_name, question))
        if qa is None:
            raise KeyError(
                f"CALC_QUESTIONS entry not found in {QA_PATH} (doc_name/question text "
                f"drifted out of sync): [{doc_name}] {question[:80]}"
            )
        resolved.append(qa)
    return resolved


def main():
    calc_pairs = load_calc_qa_pairs()
    available = _available_doc_names()
    runnable = [qa for qa in calc_pairs if qa["doc_name"] in available]
    skipped = [qa for qa in calc_pairs if qa["doc_name"] not in available]

    print(f"Calc question set: {len(calc_pairs)} total "
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
    print(f"CALC QUESTIONS RESULT: {n_pass}/{len(scored)} passed"
          + (f" ({len(results) - len(scored)} informational-only, not scored)"
             if len(results) > len(scored) else ""))
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "calc_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
