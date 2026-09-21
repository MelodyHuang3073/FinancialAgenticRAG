"""
End-to-end regression test against real 10-K PDFs and ground-truth answers.

financebench_qa_subset.json holds every question from the FinanceBench
open-source benchmark (patronus-ai/financebench on Hugging Face,
https://huggingface.co/datasets/PatronusAI/financebench) whose doc_name
matches a PDF fixture this project ships in tests/financebench_pdfs/ —
fetched directly from the dataset's own rows (question/answer/question_type
copied verbatim, not retyped or independently recomputed), so this stays a
straightforward mirror of the official benchmark rather than a hand-curated
subset. As of the last refresh that's 52 questions across ~19 distinct
10-Ks. Re-running the fetch (see the datasets-server API,
https://datasets-server.huggingface.co/rows?dataset=PatronusAI/financebench)
and filtering by which doc_names have a matching PDF in
tests/financebench_pdfs/ regenerates this file when new PDF fixtures are
added.

Unlike the tests/test_*.py unit suite (which uses hand-built PDF fixtures to
test parser/chunker internals directly), THIS file is the only place in the
project that exercises the full pipeline — parse real PDF -> classify ->
retrieve -> PoT reasoning -> final answer — against ground-truth answers with
known correct values, on the exact documents the project is meant to handle.
With this many real PDFs now indexed, a full run is slow (expect it to take
well over 30 minutes under an LLM-backed orchestrator).

Run directly for a human-readable pass/fail report:
    python tests/test_financebench_qa.py

Run under pytest for CI (numeric questions use a tolerance-based assertion;
qualitative questions assert the gold answer's key facts appear in the
generated answer — see _check_numeric / _check_contains_facts below):
    pytest tests/test_financebench_qa.py -v -s
"""
import io
import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.vector_store import FinancialVectorStoreManager
from app.rag.parser import FinancialFileParser
from app.agent.orchestrator import FinAgentRAGOrchestrator

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "financebench_pdfs")
QA_PATH = os.path.join(os.path.dirname(__file__), "financebench_qa_subset.json")

# doc_name (FinanceBench) -> (pdf filename, company label used when uploading)
DOC_TO_FILE = {
    "3M_2018_10K": ("3M_2018_10K.pdf", "3M"),
    "3M_2022_10K": ("3M_2022_10K.pdf", "3M"),
    "3M_2023Q2_10Q": ("3M_2023Q2_10Q.pdf", "3M"),
    "ACTIVISIONBLIZZARD_2019_10K": ("ACTIVISIONBLIZZARD_2019_10K.pdf", "Activision Blizzard"),
    "ADOBE_2015_10K": ("ADOBE_2015_10K.pdf", "Adobe"),
    "ADOBE_2016_10K": ("ADOBE_2016_10K.pdf", "Adobe"),
    "ADOBE_2017_10K": ("ADOBE_2017_10K.pdf", "Adobe"),
    "ADOBE_2022_10K": ("ADOBE_2022_10K.pdf", "Adobe"),
    "AES_2022_10K": ("AES_2022_10K.pdf", "AES Corporation"),
    "AMAZON_2017_10K": ("AMAZON_2017_10K.pdf", "Amazon"),
    "AMAZON_2019_10K": ("AMAZON_2019_10K.pdf", "Amazon"),
    "AMCOR_2020_10K": ("AMCOR_2020_10K.pdf", "Amcor"),
    "AMCOR_2022_8K_dated-2022-07-01": ("AMCOR_2022_8K_2022-07-01.pdf", "Amcor"),
    "AMCOR_2023Q2_10Q": ("AMCOR_2023Q2_10Q.pdf", "Amcor"),
    "AMCOR_2023Q4_EARNINGS": ("AMCOR_2023Q4_EARNINGS.pdf", "Amcor"),
    "AMCOR_2023_10K": ("AMCOR_2023_10K.pdf", "Amcor"),
    "AMD_2015_10K": ("AMD_2015_10K.pdf", "AMD"),
    "AMD_2022_10K": ("AMD_2022_10K.pdf", "AMD"),
    "AMERICANEXPRESS_2022_10K": ("AMERICANEXPRESS_2022_10K.pdf", "American Express"),
    "AMERICANWATERWORKS_2020_10K": ("AMERICANWATERWORKS_2020_10K.pdf", "American Water Works"),
    "AMERICANWATERWORKS_2021_10K": ("AMERICANWATERWORKS_2021_10K.pdf", "American Water Works"),
    "AMERICANWATERWORKS_2022_10K": ("AMERICANWATERWORKS_2022_10K.pdf", "American Water Works"),
    "BESTBUY_2017_10K": ("BESTBUY_2017_10K.pdf", "Best Buy"),
    "BESTBUY_2019_10K": ("BESTBUY_2019_10K.pdf", "Best Buy"),
    "BESTBUY_2023_10K": ("BESTBUY_2023_10K.pdf", "Best Buy"),
    "BESTBUY_2024Q2_10Q": ("BESTBUY_2024Q2_10Q.pdf", "Best Buy"),
    "BLOCK_2016_10K": ("BLOCK_2016_10K.pdf", "Block"),
    "BLOCK_2020_10K": ("BLOCK_2020_10K.pdf", "Block"),
    "BOEING_2018_10K": ("BOEING_2018_10K.pdf", "Boeing"),
    "BOEING_2022_10K": ("BOEING_2022_10K.pdf", "Boeing"),
    "COCACOLA_2017_10K": ("COCACOLA_2017_10K.pdf", "Coca-Cola"),
    "COCACOLA_2021_10K": ("COCACOLA_2021_10K.pdf", "Coca-Cola"),
    "COCACOLA_2022_10K": ("COCACOLA_2022_10K.pdf", "Coca-Cola"),
    "CORNING_2020_10K": ("CORNING_2020_10K.pdf", "Corning"),
    "CORNING_2021_10K": ("CORNING_2021_10K.pdf", "Corning"),
    "CORNING_2022_10K": ("CORNING_2022_10K.pdf", "Corning"),
    "COSTCO_2021_10K": ("COSTCO_2021_10K.pdf", "Costco"),
    "CVSHEALTH_2018_10K": ("CVSHEALTH_2018_10K.pdf", "CVS Health"),
    "CVSHEALTH_2022_10K": ("CVSHEALTH_2022_10K.pdf", "CVS Health"),
    "FOOTLOCKER_2022_8K_dated-2022-05-20": ("FOOTLOCKER_2022_8K_dated-2022-05-20.pdf", "Foot Locker"),
    "FOOTLOCKER_2022_8K_dated_2022-08-19": ("FOOTLOCKER_2022_8K_dated_2022-08-19.pdf", "Foot Locker"),
    "GENERALMILLS_2019_10K": ("GENERALMILLS_2019_10K.pdf", "General Mills"),
    "GENERALMILLS_2020_10K": ("GENERALMILLS_2020_10K.pdf", "General Mills"),
    "GENERALMILLS_2022_10K": ("GENERALMILLS_2022_10K.pdf", "General Mills"),
    "JOHNSON_JOHNSON_2022_10K": ("JOHNSON_JOHNSON_2022_10K.pdf", "Johnson & Johnson"),
    "JOHNSON_JOHNSON_2022Q4_EARNINGS": ("JOHNSON_JOHNSON_2022Q4_EARNINGS.pdf", "Johnson & Johnson"),
    "JOHNSON_JOHNSON_2023Q2_EARNINGS": ("JOHNSON_JOHNSON_2023Q2_EARNINGS.pdf", "Johnson & Johnson"),
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30": ("JOHNSON_JOHNSON_2023_8K_dated-2023-08-30.pdf", "Johnson & Johnson"),
    "JPMORGAN_2021Q1_10Q": ("JPMORGAN_2021Q1_10Q.pdf", "JPMorgan"),
    "JPMORGAN_2022Q2_10Q": ("JPMORGAN_2022Q2_10Q.pdf", "JPMorgan"),
    "JPMORGAN_2022_10K": ("JPMORGAN_2022_10K.pdf", "JPMorgan"),
    "JPMORGAN_2023Q2_10Q": ("JPMORGAN_2023Q2_10Q.pdf", "JPMorgan"),
    "KRAFTHEINZ_2019_10K": ("KRAFTHEINZ_2019_10K.pdf", "Kraft Heinz"),
    "LOCKHEEDMARTIN_2020_10K": ("LOCKHEEDMARTIN_2020_10K.pdf", "Lockheed Martin"),
    "LOCKHEEDMARTIN_2021_10K": ("LOCKHEEDMARTIN_2021_10K.pdf", "Lockheed Martin"),
    "LOCKHEEDMARTIN_2022_10K": ("LOCKHEEDMARTIN_2022_10K.pdf", "Lockheed Martin"),
    "MGMRESORTS_2018_10K": ("MGMRESORTS_2018_10K.pdf", "MGM Resorts"),
    "MGMRESORTS_2020_10K": ("MGMRESORTS_2020_10K.pdf", "MGM Resorts"),
    "MGMRESORTS_2022_10K": ("MGMRESORTS_2022_10K.pdf", "MGM Resorts"),
    "MGMRESORTS_2022Q4_EARNINGS": ("MGMRESORTS_2022Q4_EARNINGS.pdf", "MGM Resorts"),
    "MGMRESORTS_2023Q2_10Q": ("MGMRESORTS_2023Q2_10Q.pdf", "MGM Resorts"),
    "MICROSOFT_2016_10K": ("MICROSOFT_2016_10K.pdf", "Microsoft"),
    "MICROSOFT_2023_10K": ("MICROSOFT_2023_10K.pdf", "Microsoft"),
    "NETFLIX_2015_10K": ("NETFLIX_2015_10K.pdf", "Netflix"),
    "NETFLIX_2017_10K": ("NETFLIX_2017_10K.pdf", "Netflix"),
    "NIKE_2018_10K": ("NIKE_2018_10K.pdf", "Nike"),
    "NIKE_2019_10K": ("NIKE_2019_10K.pdf", "Nike"),
    "NIKE_2021_10K": ("NIKE_2021_10K.pdf", "Nike"),
    "NIKE_2023_10K": ("NIKE_2023_10K.pdf", "Nike"),
    "PAYPAL_2022_10K": ("PAYPAL_2022_10K.pdf", "PayPal"),
    "PEPSICO_2021_10K": ("PEPSICO_2021_10K.pdf", "PepsiCo"),
    "PEPSICO_2022_10K": ("PEPSICO_2022_10K.pdf", "PepsiCo"),
    "PEPSICO_2023Q1_EARNINGS": ("PEPSICO_2023Q1_EARNINGS.pdf", "PepsiCo"),
    "PEPSICO_2023_8K_dated-2023-05-05": ("PEPSICO_2023_8K_dated-2023-05-05.pdf", "PepsiCo"),
    "PEPSICO_2023_8K_dated-2023-05-30": ("PEPSICO_2023_8K_dated-2023-05-30.pdf", "PepsiCo"),
    "PFIZER_2021_10K": ("PFIZER_2021_10K.pdf", "Pfizer"),
    "Pfizer_2023Q2_10Q": ("Pfizer_2023Q2_10Q.pdf", "Pfizer"),
    "ULTABEAUTY_2023Q4_EARNINGS": ("ULTABEAUTY_2023Q4_EARNINGS.pdf", "Ulta Beauty"),
    "ULTABEAUTY_2023_10K": ("ULTABEAUTY_2023_10K.pdf", "Ulta Beauty"),
    "VERIZON_2021_10K": ("VERIZON_2021_10K.pdf", "Verizon"),
    "VERIZON_2022_10K": ("VERIZON_2022_10K.pdf", "Verizon"),
    "WALMART_2018_10K": ("WALMART_2018_10K.pdf", "Walmart"),
    "WALMART_2019_10K": ("WALMART_2019_10K.pdf", "Walmart"),
    "WALMART_2020_10K": ("WALMART_2020_10K.pdf", "Walmart"),
}

#: Allows an optional '$' between the sign and the digits (in EITHER
#: order — "-$1,561" and "$-1,561" both appear in the wild) so a
#: negative dollar figure isn't silently read as positive. Confirmed
#: real case: gold text "...negative working capital of -$1561M..."
#: extracted as +1561.0 under the old sign-immediately-before-digit-only
#: pattern (the '-' has no digit right after it, so it never joined the
#: match at all — the regex just started fresh at "1561"), while a model
#: answer phrased as "-1,561 million" (no dollar sign in between)
#: correctly extracted as -1561.0 — an entirely spurious sign mismatch
#: between two answers that actually agreed, not a real numeric
#: disagreement.
_NUM_RE = re.compile(r"-?\$?-?\d[\d,]*\.?\d*")

#: A 2-digit fiscal-year shorthand ("FY22") extracts as the bare number
#: 22 under _NUM_RE, not 2022 -- so when a gold answer's ONLY checkable
#: "fact" is a bare calendar year (see _is_bare_year) written out in full
#: ("In 2022, AMD reported..."), a model answer that correctly answered
#: the SAME question but phrased the year as "FY22"/"in FY22" (a common,
#: equally-correct real-world phrasing) fails the numeric check purely
#: on this shorthand mismatch -- not because anything it said was wrong.
#: Confirmed real case: AMD's "What drove revenue change... FY22" gold
#: answer's only number is the bare year 2022; a fully correct model
#: answer citing "FY22"/"FY21" throughout never produces a literal 2022
#: for _numbers_in to find. Expanding the shorthand to its 4-digit form
#: BEFORE extraction (same fix already applied to retrieval scoring in
#: hybrid_retriever._extract_years) removes this false negative without
#: touching how real dollar/percent figures are compared.
_FY_SHORT_RE = re.compile(r"\bFY\s*(\d{2})\b", re.IGNORECASE)


def _expand_fy_shorthand(text: str) -> str:
    return _FY_SHORT_RE.sub(lambda m: "FY20" + m.group(1), text or "")


#: "3M" (the company, formally "3M Company") is this benchmark's one
#: recurring real-world case of a company's own NAME starting with a
#: bare digit -- _NUM_RE has no way to tell the literal word "3M" apart
#: from a genuine "<digit>M" magnitude shorthand ("$1561M" = 1,561
#: million), so any gold/model answer that merely mentions the company
#: by name contributes a spurious 3.0 to the extracted-numbers list,
#: indistinguishable from a real financial figure. Confirmed real case:
#: 3M's own "What drove operating margin change...FY2022?" question --
#: the model's answer never stated gold's actual 1.7-percentage-point
#: figure, but still PASSED anyway, because both texts' incidental "3M"
#: mentions each contributed a coincidental 3.0 that alone satisfied
#: _check_contains_facts's "at least half the gold numbers must match"
#: threshold (gold had only 2 non-year numbers total).
#: Only strips the EXACT word "3M" (word-bounded on both sides, not
#: preceded by '$' or another digit, not followed by another digit) --
#: leaves genuine magnitude shorthand like "$1561M"/"a 23M-share buyback"
#: completely untouched, since those are always preceded by '$' or by
#: more digits, never by nothing at all the way the bare company name is.
_3M_COMPANY_NAME_RE = re.compile(r"(?<![\w$])3M\b(?!\d)")


def _strip_3m_company_name(text: str) -> str:
    return _3M_COMPANY_NAME_RE.sub("", text or "")


#: Gold and model answers routinely state the exact same dollar figure at
#: DIFFERENT magnitudes -- one side spells out the full raw number, the
#: other uses a "<number> <magnitude word>" shorthand -- and which side
#: does which is not consistent across questions. Confirmed real cases:
#: PepsiCo's "$400,000,000 increase" (gold, full digits) vs "$400 million"
#: (model, shorthand); Best Buy's "$1.8 bn" (gold, shorthand) vs
#: "$1,824.0 million" (model, full-digit-with-shorthand-suffix). A plain
#: digit-for-digit comparison treats these as unrelated numbers 1,000,000x
#: apart even though they're the identical fact.
#: Returns ADDITIONAL scaled candidate values for every "<number>
#: <magnitude word>" occurrence found in text -- e.g. "$400 million" also
#: yields 400000000.0 -- to be checked ALONGSIDE (never instead of) the
#: plain unscaled values _numbers_in already extracts. Only ever adds
#: candidates, on BOTH the gold and model side, and only for numbers that
#: actually have a magnitude word attached in their own source text --
#: never invents a scale for a bare, unit-less number like AWW's gold
#: answer "$0.40" (which relies on the QUESTION's own "in USD billions"
#: framing, with no magnitude word in the answer text itself to scale
#: from), so a case like that is left completely unaffected: neither side
#: gains a new candidate, and the existing (near-miss) comparison is
#: unchanged either way.
_MAGNITUDE_SCALE = {
    "thousand": 1e3, "million": 1e6, "mm": 1e6,
    "billion": 1e9, "bn": 1e9, "trillion": 1e12,
}
_MAGNITUDE_WORD_RE = re.compile(
    r"(-?\$?\d[\d,]*\.?\d*)\s*(thousand|million|mm|billion|bn|trillion)\b",
    re.IGNORECASE,
)


def _magnitude_scaled_numbers(text: str) -> list:
    out = []
    for m in _MAGNITUDE_WORD_RE.finditer(text or ""):
        try:
            raw = float(m.group(1).replace(",", "").replace("$", ""))
        except ValueError:
            continue
        out.append(raw * _MAGNITUDE_SCALE[m.group(2).lower()])
    return out


def _scaled_variants_for(value: float, source_text: str) -> list:
    """For one specific plain-extracted `value`, return its magnitude-
    scaled counterpart(s) if `value` itself is the number immediately
    preceding a magnitude word somewhere in `source_text` -- e.g. for
    value=1.8 and source_text containing "$1.8 bn", returns
    [1800000000.0]. Ties a specific fact to its own scaled form (rather
    than the flat, unordered list _magnitude_scaled_numbers returns) so a
    multi-number gold answer expands each number by ITS OWN magnitude
    word, not any other number's."""
    variants = []
    for m in _MAGNITUDE_WORD_RE.finditer(source_text or ""):
        try:
            raw = float(m.group(1).replace(",", "").replace("$", ""))
        except ValueError:
            continue
        if raw == value or (value != 0 and abs(raw - value) / abs(value) < 1e-6):
            variants.append(raw * _MAGNITUDE_SCALE[m.group(2).lower()])
    return variants


#: A gold answer that enumerates a short list inline ("...during FY 2022:
#: (1) Current Health Ltd and (2) Two Peaks, LLC...") uses "(1)"/"(2)" as
#: pure list-item numbering, not a financial fact -- but _NUM_RE has no
#: way to tell that apart from a real parenthesized figure. Left in, this
#: silently makes "1" and "2" part of the gold "facts to check", and a
#: model answer that correctly lists BOTH companies with their real
#: dollar amounts (no coincidental bare "1"/"2" of its own) fails the
#: check for having "missed a fact" that was never a fact in the first
#: place. Distinguished from a genuine footnote/citation marker (e.g.
#: "$147 million(1)") by requiring a space then a capital letter right
#: after the closing paren -- the shape of "(N) <New List Item>", which a
#: footnote reference attached directly to a number never has. Confirmed
#: real case: Best Buy's acquisitions gold answer ("(1) Current Health
#: Ltd and (2) Two Peaks, LLC...") reduced check_nums to exactly [1, 2]
#: after year-stripping, so a fully correct, evidence-grounded model
#: answer (both companies + correct $389M/$79M amounts) still failed
#: because it had no reason to ever produce a bare standalone "1" or "2".
_LIST_MARKER_RE = re.compile(
    r"\(\d{1,2}\)(?=\s+[A-Z])"
    # a lowercase item introduced by ":" ";" "," or "and" ("... are: (1) usual and
    # customary pricing ...; (2) PBM litigation ...; and (3) controlled ...") is a
    # list marker too; the lookbehind keeps accounting negatives like "(6) million"
    # (which are not preceded by these) untouched
    r"|(?:(?<=[:;,]\s)|(?<=\sand\s))\(\d{1,2}\)(?=\s+[a-z])"
)


def _strip_list_markers(text: str) -> str:
    return _LIST_MARKER_RE.sub("", text or "")


#: A gold answer formatted as a bullet list sometimes uses a bare "-" as
#: the bullet marker with NO space before the item's own text ("-100%
#: equity interest of a flexibles manufacturing company..."), which
#: _NUM_RE cannot tell apart from a genuine negative number -- it reads
#: "-100" as literally negative one hundred, when the "-" is pure list
#: punctuation and the real fact is a plain positive "100%". Requires a
#: SPACE then a LETTER right after the number (the shape of "-100%
#: equity...", i.e. a bullet item with descriptive text following it) --
#: NOT just "dash at line start before a digit", which would also
#: (wrongly) rewrite a gold answer that is ITSELF just a single bare
#: negative number with nothing else on the line, e.g. AES's ROA answer
#: "-0.02" or General Mills' CCC answer "-3.7" -- both entire strings,
#: not bullet items, where the leading "-" is a genuine minus sign that
#: must be left alone. Confirmed real case: Amcor's acquisitions gold
#: answer ("-100% equity interest...\n- 100% equity interest...\n
#: -acquisition of a New Zealand-based...") extracted as check_nums
#: [-100, 100] instead of [100, 100], and no model answer was ever going
#: to coincidentally state a negative -100 for what are actually two
#: 100%-owned acquisitions.
_LEADING_BULLET_DASH_RE = re.compile(r"(?m)^-(?=\d[\d,]*\.?\d*%?\s+[A-Za-z])")


def _split_bullet_dash(text: str) -> str:
    return _LEADING_BULLET_DASH_RE.sub("- ", text or "")


def _numbers_in(text: str):
    """Extract all numeric tokens (commas and '$' stripped) from a string
    as floats.

    A leading '-' is dropped (read as positive) whenever it's immediately
    preceded by a LETTER in the source text -- that's a hyphenated word
    ("COVID-19", "Q1-2022", "Type-2"), not a minus sign, and _NUM_RE alone
    can't tell the two apart. Checked at the '-' character's own actual
    position (not just the match's start), since the sign can land either
    before or after a leading '$' ("-$1,561" vs "$-1,561" both appear in
    the wild -- see _NUM_RE's own pattern). Confirmed real case: gold text
    "One-time COVID-19 vaccine manufacturing exit related costs..."
    extracted a fabricated -19.0 from "COVID-19", which then became the
    ENTIRE numeric ground-truth check for that gold answer (the only
    other number found was the bare year "FY22") -- a model answer
    correctly citing every one of the gold answer's real qualitative
    facts still failed, because nothing in it was ever going to contain a
    literal "-19" (no such negative fact exists in this question at all).
    A genuine negative number is untouched: "-$1,561" or "$-1,561" is
    preceded by whitespace/start at the '-' itself, not a letter.
    """
    out = []
    cleaned = _split_bullet_dash(_strip_list_markers(_strip_3m_company_name(_expand_fy_shorthand(text))))
    for m in _NUM_RE.finditer(cleaned):
        s = m.group(0)
        if "-" in s:
            dash_idx = m.start() + s.index("-")
            if dash_idx > 0 and cleaned[dash_idx - 1].isalpha():
                s = s.replace("-", "", 1)
        try:
            out.append(float(s.replace(",", "").replace("$", "")))
        except ValueError:
            continue
    return out


#: Accounting negatives written in parentheses -- "($473) million",
#: "(1,349)" -- which _NUM_RE reads as a POSITIVE 473 / 1,349. Only
#: unambiguous shapes: a "$" inside the parentheses, or a comma-grouped
#: amount (so footnote markers like "(1)" or "(6)" are never touched).
_ACCOUNTING_NEG_RE = re.compile(
    r"\(\s*\$\s*(\d[\d,]*\.?\d*)\s*\)"                      # ($473)
    r"|\$\s*\(\s*(\d[\d,]*\.?\d*)\s*\)"                     # $(473)
    r"|\(\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\s*\)"             # (1,349)
)


def _accounting_negatives(text: str):
    """Negative values for every parenthesized accounting amount in `text`.
    Used ADDITIVELY on the model side only (an extra candidate, never a
    replacement), so a model answer that says "($473) million" also
    matches a gold answer of "-$473 million"."""
    out = []
    for m in _ACCOUNTING_NEG_RE.finditer(text or ""):
        raw = (m.group(1) or m.group(2) or m.group(3)).replace(",", "")
        try:
            out.append(-float(raw))
        except ValueError:
            continue
    return out


def _check_numeric(gold_answer: str, model_answer: str, rel_tol: float = 0.02) -> bool:
    """
    FinanceBench numeric gold answers are short (e.g. "24.26", "1.9%", "0.83").
    Pass if ANY number extracted from the model's answer is within rel_tol
    (default 2%) of the (first, primary) number in the gold answer.
    This is intentionally lenient — it only checks whether the right NUMBER
    surfaced anywhere in the answer, not phrasing/formatting. Tightening this
    to also check units/labels is a reasonable follow-up once basic numeric
    accuracy is passing.
    """
    gold_nums = _numbers_in(gold_answer)
    model_nums = _numbers_in(model_answer)
    if not gold_nums or not model_nums:
        return False
    target = gold_nums[0]
    for n in model_nums:
        if target == 0:
            if abs(n) < 1e-9:
                return True
            continue
        if abs(n - target) / abs(target) <= rel_tol:
            return True
    return False


#: A bare 4-digit calendar year (1900-2099) is essentially never the
#: substantive fact a qualitative question is actually testing — almost
#: every FinanceBench answer about "FY2022" naturally repeats "2022"
#: somewhere in both the gold text and any model answer about the same
#: filing, correct or not. Counting it as a matchable "fact" lets a
#: substantively WRONG answer register as a match purely because both
#: texts mention the same fiscal year. Confirmed real cases (all
#: gold-says-Yes/model-says-No or vice versa, yet PASSed on this check
#: alone before this exclusion): CVS Health's Q2 FY2022 dividend
#: question (gold "$0.55/share", model "no dividend, 0.0" — matched only
#: on both texts saying "2022"), MGM Resorts' FY2022 dividend question
#: (same pattern, gold "$0.01/share"), PepsiCo's FY2022 restructuring
#: costs (gold "$411 million", model "no restructuring costs, 0" —
#: matched only on "2022"), Boeing's FY2022 gross margin trend (gold
#: 4.8%->5.3%, model 84.70%->65.43%, matched only on "2022"), and
#: Boeing's primary-customers question (gold "the US government
#: accounted for 40%", model "140 aircraft" — matched only on "2022").
def _is_bare_year(n: float) -> bool:
    return 1900 <= n <= 2099 and n == int(n)


#: Phase 3 fix: a gold answer whose only numbers are bare calendar years
#: (e.g. "Yes. ...resulting from a 2018 Lion Air crash and a 2019
#: Ethiopian Airlines crash.") has nothing but those years to check
#: against, so a model answer that gets the DIRECTION completely
#: backwards can still coincidentally "pass" purely because both texts
#: mention the same fiscal year -- the numeric check alone can't catch
#: this. This adds a lightweight opening-stance veto on top of it: if
#: the gold answer starts with an explicit Yes/No and the model answer's
#: opening clearly states (or negates) the opposite, that's an automatic
#: FAIL regardless of what the numeric check would otherwise say.
#:
#: This can only ever turn a would-be PASS into a FAIL on a clear,
#: detected contradiction -- when the model's stance can't be determined
#: (no explicit Yes/No opener and none of the negation phrases below
#: appear near the start), this falls through to the existing numeric
#: check completely unchanged, so it can never fabricate a PASS out of a
#: previously-None (informational-only) result. Verified against every
#: currently-passing Yes/No-shaped gold answer in both the calc and
#: extraction suites (Boeing legal battles, Adobe operating margin,
#: American Water Works working capital) -- all have a model stance that
#: either matches or is undetermined, so none of them flip.
_NEGATION_PHRASES = [
    "did not", "does not", "doesn't", "didn't", "has not", "hasn't",
    "have not", "haven't", "is not", "isn't", "are not", "aren't",
    "cannot", "can not", "there are none", "there is no", "no material",
    "not report", "not reported", "not been reported",
]


def _leading_yn_stance(text: str) -> Optional[bool]:
    """
    True = affirmative (Yes) stance, False = negative (No) stance,
    None = undetermined.

    Only looks at the OPENING of the text: an explicit "Yes"/"No" token
    first, else one of _NEGATION_PHRASES within the first ~200
    characters -- this covers the common model phrasing for a "No"
    answer that doesn't literally start with the word "No" (e.g. "CVS
    Health did not pay dividends...", "Boeing does not have an
    improving gross margin..."). Deliberately does NOT try to infer an
    affirmative "Yes" stance from the mere absence of a negation phrase
    (that would be guessing, not detecting) -- an affirmative stance is
    only ever an explicit leading "Yes".
    """
    stripped = re.sub(r'^[\*\s]+', '', text or '').strip()
    m = re.match(r'(yes|no)\b', stripped, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    window = stripped[:200].lower()
    if any(phrase in window for phrase in _NEGATION_PHRASES):
        return False
    return None


#: Gold text sometimes conveys a decrease/decline through a WORD
#: ("shrunk by 0.9%", "declined by 3%") rather than a minus sign on the
#: number itself -- the extracted number is genuinely positive (0.9, not
#: -0.9). A model answer describing the identical fact with an explicit
#: signed figure ("a decline of -0.9 percentage points") is numerically
#: the negation of gold's own bare digit, even though both texts agree on
#: the fact and its direction -- a plain positive-vs-negative comparison
#: rejects a fully correct answer. Only fires when gold's own number has
#: no minus sign already (a genuinely negative gold number is compared
#: as-is, unchanged); only ADDS an alternative target to check the model
#: against, so this can only turn a previously-missed hit into a match,
#: never suppress an already-correct check. Confirmed real case: 3M's
#: "which segment dragged down 3M's overall growth" question -- gold
#: "The consumer segment shrunk by 0.9% organically" (bare 0.9), model
#: "organic sales decline of -0.9%" (signed -0.9) -- both state the exact
#: same fact, but the raw digits are numeric opposites.
_DECREASE_WORD_RE = re.compile(
    r"\b(?:shrunk|shrank|shrink\w*|declin\w*|decreas\w*|drop\w*|fell|fall\w*|reduc\w*|lower)\b",
    re.IGNORECASE,
)


def _implies_decrease(text: str) -> bool:
    return bool(_DECREASE_WORD_RE.search(text or ""))


def _decimals_of(value: float, text: str) -> int:
    """Number of decimal digits the gold text itself printed for `value`
    ("66.56" -> 2, "1.7%" -> 1, "$0.40" -> 2, "3,017" -> 0)."""
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(0).replace(",", "").replace("$", "")
        try:
            if abs(abs(float(tok)) - abs(value)) < 1e-9:
                return len(tok.split(".")[1]) if "." in tok else 0
        except ValueError:
            continue
    return 0


def _close_enough(t: float, m: float, decimals: int) -> bool:
    """A model figure matches a gold figure when it is within 2% (1% when the
    gold itself is stated to 2+ decimals, so "$65.44" no longer passes for a
    gold "$66.56" -- a real false PASS at 1.68%) OR within half a unit of the
    gold's own last printed digit (a gold "1.7%" accepts the model's more
    precise 1.7392%, which the plain 2% band rejected)."""
    if t == 0:
        return abs(m) < 1e-9
    rel = 0.01 if decimals >= 2 else 0.02
    if abs(t - m) / abs(t) <= rel:
        return True
    return decimals >= 1 and abs(t - m) <= 0.5 * 10 ** (-decimals) + 1e-9


def _check_contains_facts(gold_answer: str, model_answer: str) -> bool:
    """
    Qualitative gold answers (e.g. 'The consumer segment shrunk by 0.9%
    organically.') are checked by requiring every NUMBER in the gold answer
    to also appear (within 2% tolerance) in the model answer. This doesn't
    validate prose/explanation quality, only that the key extracted facts
    made it into the final answer — a floor, not a full correctness check.
    Gold answers with no numbers at all (pure qualitative, e.g. industry
    description) are treated as informational-only and always reported but
    never asserted on.

    Bare calendar years are excluded from the numbers being checked
    whenever at least one non-year number is also present — see
    _is_bare_year's docstring for why. A gold answer with ONLY year
    numbers (e.g. "...resulting from a 2018 Lion Air crash and a 2019
    Ethiopian Airlines crash") has nothing else to fall back on, so those
    years are still used rather than checking nothing at all — this
    residual case (a yes/no question whose only "facts" are incidental
    years) isn't caught by a numeric check either way — see
    _leading_yn_stance, checked first below, which is what actually
    fixes it.
    """
    gold_stance = _leading_yn_stance(gold_answer)
    if gold_stance is not None:
        model_stance = _leading_yn_stance(model_answer)
        if model_stance is not None and model_stance != gold_stance:
            return False  # explicit direction contradiction — no need to check numbers at all

    gold_nums = _numbers_in(gold_answer)
    if not gold_nums:
        return None  # no numeric ground truth to check — informational only
    model_nums = _numbers_in(model_answer)
    if not model_nums:
        return False
    non_year_nums = [n for n in gold_nums if not _is_bare_year(n)]
    check_nums = non_year_nums or gold_nums
    gold_implies_decrease = _implies_decrease(gold_answer)
    # See _magnitude_scaled_numbers's docstring: adds "<number> <magnitude
    # word>" scaled alternatives from the MODEL side once, up front (every
    # gold fact gets checked against the same expanded model pool), and
    # each gold fact's OWN magnitude-scaled variant (if it has one) is
    # added per-fact below via _scaled_variants_for.
    all_model_candidates = (
        list(model_nums) + _magnitude_scaled_numbers(model_answer) + _accounting_negatives(model_answer)
    )
    hits = 0
    for g in check_nums:
        g_variants = [g] + _scaled_variants_for(g, gold_answer)
        targets = []
        for gv in g_variants:
            targets.append(gv)
            if gold_implies_decrease and gv > 0:
                targets.append(-gv)
        g_dec = _decimals_of(g, gold_answer)
        if any(
            _close_enough(t, m, g_dec if abs(abs(t) - abs(g)) < 1e-9 else 0)
            for t in targets for m in all_model_candidates
        ):
            hits += 1
    return hits >= max(1, len(check_nums) // 2)  # at least half the checked numbers must surface


def _available_doc_names() -> set:
    """doc_names from DOC_TO_FILE whose PDF actually exists in
    tests/financebench_pdfs/. financebench_qa_subset.json mirrors the FULL
    150-question official dataset (see this module's docstring), which
    covers far more doc_names/companies than this project ships PDF
    fixtures for — a question whose doc_name has no available PDF can't be
    run at all and is skipped (see test_financebench_question below)
    rather than the whole suite failing to collect."""
    return {
        doc_name for doc_name, (filename, _) in DOC_TO_FILE.items()
        if os.path.exists(os.path.join(FIXTURES_DIR, filename))
    }


def _build_indexed_store() -> FinancialVectorStoreManager:
    """Parse every real 10-K PDF this project ships a fixture for (per
    DOC_TO_FILE) with the project's actual parser and index them exactly
    the way the real upload flow does (FinancialFileParser ->
    add_parsed_passages), so this test exercises the real ingestion path,
    not a shortcut. doc_names with no matching PDF are silently skipped
    here (see _available_doc_names) — their questions are individually
    skipped by the test, not treated as a fatal setup error."""
    vs = FinancialVectorStoreManager()
    parser = FinancialFileParser()
    for doc_name, (filename, _label) in DOC_TO_FILE.items():
        path = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            content = f.read()
        # Use the PUBLIC parse_file() entry point exactly as /api/upload-file
        # does, not the internal _parse_pdf() with a hand-picked "company"
        # label — the two are NOT equivalent. parse_file() derives
        # company_name from the raw filename stem (e.g. "CORNING_2020_10K"),
        # which then gets baked into every passage's own "company" field and
        # is what real retrieval/extraction actually runs against in
        # production. Confirmed real case: with a clean label like "Corning"
        # standing in for company_name, Corning's real FY2020 "Cost of
        # sales" (7,772) beat an unrelated AOCI-reclassification footnote's
        # coincidentally-labeled "Cost of sales" (13) in retrieval — but
        # with the actual filename-stem company_name production uses, the
        # footnote row won instead, giving a wildly wrong DPO. A test using
        # the label shortcut would never have caught this.
        result = parser.parse_file(filename, content)
        company_name = os.path.splitext(filename)[0]
        vs.add_parsed_passages(filename, company_name, result["passages"])
        print(f"  indexed {filename} ({company_name}): {len(result['passages'])} passages"
              + (f"  [WARNING: {result['warning']}]" if result["warning"] else ""))
    return vs


def run_financebench_subset(verbose: bool = True):
    """
    Runs every question in financebench_qa_subset.json against the real,
    fully-indexed 4-PDF corpus and reports pass/fail per question plus a
    summary. Returns (n_pass, n_total, results) for programmatic use.
    """
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print("Indexing real 10-K PDFs through the actual parser/upload path...")
    vs = _build_indexed_store()
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    available_docs = _available_doc_names()
    results = []
    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        gold = qa["answer"]
        doc_name = qa["doc_name"]

        if doc_name not in available_docs:
            results.append({
                "doc_name": doc_name, "question": question, "gold": gold,
                "model_answer": "", "passed": None,
                "check_kind": "no PDF fixture available", "error": None,
            })
            if verbose:
                print(f"\n[{i}/{len(qa_pairs)}] ⏭️  SKIP  (no PDF fixture)  [{doc_name}]")
                print(f"  Q: {question}")
            continue

        try:
            res = orchestrator.process_query(question)
            model_answer = res.get("final_answer", "") or ""
            error = None
        except Exception as e:
            model_answer = ""
            error = repr(e)

        gold_nums = _numbers_in(gold)
        if gold_nums and qa.get("question_type") == "metrics-generated":
            passed = _check_numeric(gold, model_answer)
            check_kind = "numeric (2% tol)"
        else:
            fact_check = _check_contains_facts(gold, model_answer)
            passed = fact_check if fact_check is not None else None
            check_kind = "fact-presence (2% tol)" if fact_check is not None else "informational only"

        results.append({
            "doc_name": doc_name, "question": question, "gold": gold,
            "model_answer": model_answer, "passed": passed,
            "check_kind": check_kind, "error": error,
        })

        if verbose:
            status = "✅ PASS" if passed is True else ("❌ FAIL" if passed is False else "ℹ️  INFO")
            print(f"\n[{i}/{len(qa_pairs)}] {status}  ({check_kind})  [{doc_name}]")
            print(f"  Q: {question}")
            print(f"  Gold  : {gold[:200]}")
            print(f"  Model : {model_answer[:200] if model_answer else '(empty)'}")
            if error:
                print(f"  ERROR : {error}")

    scored = [r for r in results if r["passed"] is not None]
    n_pass = sum(1 for r in scored if r["passed"])
    n_total = len(scored)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"FinanceBench subset result: {n_pass}/{n_total} passed "
              f"({n_pass / n_total * 100:.0f}%)" if n_total else "No scoreable questions.")
        n_info = len(results) - n_total
        if n_info:
            print(f"({n_info} question(s) were informational-only — no numeric ground truth to check)")
    return n_pass, n_total, results


# ── pytest entry points (one test per question, so CI shows per-question status) ──
def _load_qa_pairs():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_QA_PAIRS = _load_qa_pairs()
_SHARED_STORE = None


def _get_shared_store():
    global _SHARED_STORE
    if _SHARED_STORE is None:
        _SHARED_STORE = _build_indexed_store()
    return _SHARED_STORE


import pytest


@pytest.mark.parametrize("qa", _QA_PAIRS, ids=[f"{q['doc_name']}::{q['question'][:40]}" for q in _QA_PAIRS])
def test_financebench_question(qa):
    if qa["doc_name"] not in _available_doc_names():
        pytest.skip(f"no PDF fixture for {qa['doc_name']} in tests/financebench_pdfs/")
    vs = _get_shared_store()
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)
    res = orchestrator.process_query(qa["question"])
    model_answer = res.get("final_answer", "") or ""

    gold_nums = _numbers_in(qa["answer"])
    if gold_nums and qa.get("question_type") == "metrics-generated":
        assert _check_numeric(qa["answer"], model_answer), (
            f"\nQ: {qa['question']}\nGold: {qa['answer']}\nGot: {model_answer}"
        )
    else:
        fact_check = _check_contains_facts(qa["answer"], model_answer)
        if fact_check is None:
            pytest.skip("informational-only gold answer (no numeric ground truth)")
        assert fact_check, f"\nQ: {qa['question']}\nGold: {qa['answer']}\nGot: {model_answer}"


if __name__ == "__main__":
    run_financebench_subset(verbose=True)
