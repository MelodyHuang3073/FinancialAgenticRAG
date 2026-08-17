---
name: financebench-handler
description: >
  Four-stage SOP for answering FinanceBench-style financial report questions.
  Covers intent classification, block targeting, multi-step RAG retrieval, and
  mandatory Python-based calculation with source attribution.
type: rag-reasoning-skill
version: "1.0.0"
author: FinAgent-RAG Team
tags:
  - financebench
  - financial-analysis
  - rag
  - program-of-thought
  - 10-K
  - SEC
---

# FinanceBench Handler — Financial Q&A SOP

> **Activate this Skill** when the user asks any question that involves:
> numerical values from financial statements, year-over-year comparisons,
> margin / ratio calculations, or qualitative reasoning from MD&A / Footnotes.

---

## Overview

Financial report questions cannot be reliably answered by a single RAG pass.
This Skill enforces a **4-stage pipeline** that mirrors how a professional
financial analyst works:

```
[Question]
   |
   v Stage 1 — Intent Classification
   |
   v Stage 2 — Block Targeting
   |
   v Stage 3 — Multi-Step Retrieval  (calls rag_retriever.py)
   |
   v Stage 4 — Calculation & Verification  (generates Python code)
   |
[Answer + Source Citation]
```

---

## Stage 1 — Intent Classification

Classify the question into **exactly one** of the following four types before
taking any other action.

| Type | Label | Trigger Patterns |
|------|-------|-----------------|
| Single data point | `FACT_SINGLE` | "What was X in year Y?", "How much is …" |
| Cross-period comparison | `FACT_MULTI` | "Compare … between Y1 and Y2", "How did X change?" |
| Ratio / formula calculation | `CALCULATION` | "YoY growth", "gross margin", "CAGR", "ROE", "EPS" |
| Qualitative reasoning | `REASONING` | "Why did … change?", "What drove …", "Explain …" |

### Classification Rules

1. **If the question contains TWO or more years** -> FACT_MULTI or CALCULATION.
2. **If the question asks for a percentage, rate, or ratio** -> CALCULATION.
3. **If the question asks for causes, drivers, or management commentary** -> REASONING.
4. **Otherwise** -> FACT_SINGLE.

> Output the label explicitly, e.g.: `[Intent: CALCULATION]`

> **Do NOT proceed** to Stage 2 until the intent type is confirmed.

---

## Stage 2 — Block Targeting

Map the intent type to the correct section of the financial report.

### 2.1 Financial Statement Blocks (Three-Statement Model)

| Target Block | Use When | Key Line Items |
|---|---|---|
| **Income Statement** | Revenue, Gross Profit, Operating Income, Net Income, EPS, Margin % | Net sales, COGS, R&D, SG&A |
| **Balance Sheet** | Assets, Liabilities, Equity, Debt, Working Capital | Total assets, Long-term debt, Stockholders' equity |
| **Cash Flow Statement** | Operating / Investing / Financing cash flows, CapEx, FCF | Operating CF, Capital expenditures |

### 2.2 Narrative Blocks

| Target Block | Use When |
|---|---|
| **MD&A** (Management Discussion & Analysis) | `REASONING` — causes, drivers, strategic context |
| **Footnotes / Notes to Financial Statements** | Accounting policy, segment breakdown, lease obligations, tax details |
| **Risk Factors** | Questions about business risk, regulatory exposure |

### 2.3 Block Targeting Decision Table

```
FACT_SINGLE   -> Primary: Statement block matching the metric
FACT_MULTI    -> Primary: Same statement block, two year columns
CALCULATION   -> Primary: Statement block(s) needed for numerator & denominator
REASONING     -> Primary: MD&A  |  Secondary: Statement block for numbers
```

> Always retrieve from the MOST SPECIFIC block first. If insufficient data,
> escalate to a wider block.

---

## Stage 3 — Multi-Step Retrieval

### 3.1 Decompose Complex Questions

Before calling the retriever, decompose `CALCULATION` and `REASONING`
questions into atomic sub-queries:

**Example:**
```
Question: "What was 3M's gross margin in 2022 vs 2021, and why did it change?"

Sub-queries:
  Q1: "3M gross profit 2022"          -> Income Statement
  Q2: "3M net sales (revenue) 2022"   -> Income Statement
  Q3: "3M gross profit 2021"          -> Income Statement
  Q4: "3M net sales (revenue) 2021"   -> Income Statement
  Q5: "3M gross margin change reason" -> MD&A
```

### 3.2 Calling rag_retriever.py (Parent-Child RAG)

Use the `ParentChildRetriever` for precise chunk retrieval:

```python
from rag_retriever import ParentChildRetriever

retriever = ParentChildRetriever(db_json_path="global_library_db.json")

# Structured table search
results = retriever.search_by_metadata(
    key_term="gross profit",
    chapter="income statement",
    chunk_type="table_row",
    book="3M_2022_10K"
)

# Semantic search with parent context
results = retriever.semantic_search(
    query="gross margin decline drivers 2022",
    top_k=5,
    return_parent=True   # return full parent paragraph for REASONING
)
```

### 3.3 Retrieval Quality Gates

Before proceeding to Stage 4, verify ALL of the following:

- [ ] Retrieved chunk contains the **company name** matching the question
- [ ] Retrieved chunk covers the **correct fiscal year(s)**
- [ ] The **unit** of the value is confirmed (Thousands / Millions / Billions)
- [ ] For `CALCULATION`: both **numerator** and **denominator** values are retrieved
- [ ] For `FACT_MULTI`: values for **all required years** are present

> Do NOT compute using values that fail any quality gate.
> Re-query with a refined sub-query instead.

---

## Stage 4 — Calculation & Verification

### 4.1 Mandatory Python Code Generation

> **LLM mental arithmetic is STRICTLY PROHIBITED.**
>
> All numerical computations MUST be performed by generated Python code
> executed in a sandboxed environment. This eliminates rounding errors and
> hallucinated calculations.

#### Standard Calculation Templates

```python
# ── YoY Growth Rate ──────────────────────────────────────────────
revenue_2022 = 34_229   # Net sales, 3M, FY2022 ($ millions)
revenue_2021 = 35_355   # Net sales, 3M, FY2021 ($ millions)
yoy = (revenue_2022 - revenue_2021) / revenue_2021 * 100
print("Revenue YoY Growth (2021 to 2022): %.2f%%" % yoy)
# Output: Revenue YoY Growth (2021 to 2022): -3.18%

# ── Gross Margin ─────────────────────────────────────────────────
gross_profit = 12_523   # $ millions
net_sales    = 34_229   # $ millions
gross_margin = gross_profit / net_sales * 100
print("Gross Margin: %.2f%%" % gross_margin)

# ── CAGR ─────────────────────────────────────────────────────────
start_val = 28_200   # Revenue FY2019
end_val   = 34_229   # Revenue FY2022
n_years   = 3
cagr = (end_val / start_val) ** (1 / n_years) - 1
print("3-Year Revenue CAGR: %.2f%%" % (cagr * 100))

# ── ROE ──────────────────────────────────────────────────────────
net_income = 5_777   # $ millions
avg_equity = 4_623   # $ millions (average of start & end period)
roe = net_income / avg_equity * 100
print("ROE: %.2f%%" % roe)
```

### 4.2 Unit Normalisation (Critical)

Before assigning any value to a variable, confirm and normalise units:

```python
# Source: "In millions, except per-share amounts"
# All values already in $ millions; no conversion needed.

# If source says "In thousands":
value_millions = value_thousands / 1_000

# If mixing Billions with Millions:
revenue_B = 34.229   # stated as $34.229B
revenue_M = revenue_B * 1_000  # -> 34,229 $ millions
```

### 4.3 Sanity Range Check

| Metric | Expected Range |
|---|---|
| Gross Margin (manufacturing) | 30 – 65 % |
| Net Profit Margin (S&P 500 avg) | 5 – 20 % |
| Revenue YoY (stable large-cap) | -15 % to +30 % |
| ROE (healthy large-cap) | 10 – 30 % |
| Current Ratio (healthy) | 1.0 – 3.0 x |

If the result falls **outside the expected range**, re-check:
1. Was the correct year column selected?
2. Are units consistent across numerator and denominator?
3. Is the line item unambiguous (e.g., "Net sales" vs "Gross revenue")?

### 4.4 Source Attribution Format

Every final answer MUST include source citations:

```
Answer: 3M's revenue YoY growth in 2022 was **-3.18%**.

Sources:
- Net sales 2022: $34,229M — 3M 2022 Annual Report (10-K),
  Consolidated Statements of Income, Page 57
- Net sales 2021: $35,355M — 3M 2022 Annual Report (10-K),
  Consolidated Statements of Income, Page 57
- Formula: (34,229 - 35,355) / 35,355 x 100 = -3.18%
```

---

## Quick Reference — Common FinanceBench Formula Map

| Question Pattern | Formula | Required Variables |
|---|---|---|
| YoY Growth | (V_new - V_old) / V_old x 100 | metric value for 2 years |
| CAGR | (V_end / V_start)^(1/n) - 1 | start val, end val, n years |
| Gross Margin | Gross Profit / Net Sales x 100 | gross profit, net sales |
| Operating Margin | Operating Income / Net Sales x 100 | op. income, net sales |
| Net Margin | Net Income / Net Sales x 100 | net income, net sales |
| ROE | Net Income / Avg. Shareholders Equity x 100 | net income, equity |
| ROA | Net Income / Avg. Total Assets x 100 | net income, total assets |
| Current Ratio | Current Assets / Current Liabilities | CA, CL |
| EPS (Diluted) | Net Income / Diluted Shares Outstanding | net income, diluted shares |
| FCF | Operating Cash Flow - CapEx | OCF, capex |
| Debt-to-Equity | Total Debt / Shareholders Equity | total debt, equity |

---

## Error Handling

| Error Scenario | Required Action |
|---|---|
| Retrieved chunk is from wrong company | Re-query with explicit company name filter |
| Year column is missing from chunk | Widen retrieval window; search parent chunk |
| Value is 0 or negative unexpectedly | Check if item is a loss or contra-account |
| Unit mismatch detected | Re-read source footnotes; normalise explicitly |
| Result fails sanity range check | Re-verify data sources before re-computing |
| No relevant chunks retrieved | Decompose query further; try alternate keywords |

---

## Activation Checklist

Before answering any FinanceBench question, confirm:

- [ ] **Stage 1**: Intent classified (FACT_SINGLE / FACT_MULTI / CALCULATION / REASONING)
- [ ] **Stage 2**: Target financial report block(s) identified
- [ ] **Stage 3**: All required sub-queries issued; quality gates passed
- [ ] **Stage 4**: Python code generated and executed; units verified; result within sanity range
- [ ] **Citation**: Every number in the answer is traced to a specific source chunk
