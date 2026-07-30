import re
from typing import List, Dict, Any
from app.tools.sandbox import execute_pot_code

class ProgramOfThoughtReasoner:
    """
    Program-of-Thought (PoT) Financial Reasoner (FinAgent-RAG Paper Section 3.4)
    Generates executable Python code for numerical financial computations instead of LLM mental arithmetic.
    """

    def generate_and_execute(self, query: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_lower = query.lower()

        extracted_vars = {}

        # 1. Variable extraction: Two-phase approach
        for ev in evidence_list:
            content = ev.get("content", "")
            if not content:
                continue

            # Phase 1: Split content by '|' to get individual fields
            fields = [f.strip() for f in content.split('|')]

            # Extract Line Item from the 'Line Item:' field if available
            line_item_name = None
            for f in fields:
                if f.startswith("Line Item:"):
                    line_item_name = f.split("Line Item:", 1)[1].strip()
                    break

            # Phase 2: For each field, try to match header: value pattern using regex
            for f in fields:
                match = re.search(r'(.*(?:20\d{2}|FY\d{4})[^:]*?)\s*:\s*([\d,]+\.?\d*)', f)
                if match:
                    year_header = match.group(1).strip()
                    val_str = match.group(2).strip()

                    year_match = re.search(r'(20\d{2}|FY\d{4})', year_header)
                    if not year_match:
                        continue
                    year = year_match.group(1)

                    try:
                        val_clean = float(val_str.replace(',', ''))
                    except ValueError:
                        continue

                    item_name = line_item_name if line_item_name else year_header
                    # Sanitize item names with re.sub(r'\W+', '_', item).lower()[:20]
                    sanitized_item = re.sub(r'\W+', '_', item_name).strip('_').lower()[:20]
                    if not sanitized_item:
                        sanitized_item = "item"

                    var_key = f"val_{year}_{sanitized_item}"

                    dedup_key = var_key
                    dup_idx = 1
                    while dedup_key in extracted_vars and extracted_vars[dedup_key]["val"] != val_clean:
                        dedup_key = f"val_{year}_{sanitized_item}_{dup_idx}"
                        dup_idx += 1

                    if dedup_key not in extracted_vars:
                        extracted_vars[dedup_key] = {
                            "item": item_name,
                            "year": year,
                            "val": val_clean,
                            "code_key": dedup_key
                        }

        # Fallback: If no variables can be extracted at all (e.g., unstructured text)
        if not extracted_vars:
            raw_text_combined = "\n".join([ev.get("content", "") for ev in evidence_list])
            numbers = re.findall(r'([\d,]+\.\d+)', raw_text_combined)
            if not numbers:
                numbers = re.findall(r'([\d,]+\.?\d+)', raw_text_combined)
            for idx, n_str in enumerate(numbers[:6], 1):
                try:
                    num_val = float(n_str.replace(',', ''))
                    var_key = f"num_{idx}"
                    extracted_vars[var_key] = {
                        "item": f"Value_{idx}",
                        "year": "N/A",
                        "val": num_val,
                        "code_key": var_key
                    }
                except ValueError:
                    pass

        code_lines = ["# FinAgent-RAG Program-of-Thought (PoT) Sandbox Script"]
        code_lines.append("# Step 1: Extracted numerical variables from retrieved financial report")

        for k, v in extracted_vars.items():
            if v["year"] != "N/A":
                code_lines.append(f"{v['code_key']} = {v['val']}  # {v['item']} ({v['year']})")
            else:
                code_lines.append(f"{v['code_key']} = {v['val']}")

        code_lines.append("\n# Step 2: Deterministic Financial Calculation")

        target_item_keywords = []
        if "營業收入" in query or "revenue" in q_lower:
            target_item_keywords.extend(["營業收入", "revenue"])
        elif "營業費用" in query or "operating expense" in q_lower:
            target_item_keywords.extend(["營業費用", "operating expense", "operating expenses"])
        elif "研發費用" in query or "r&d" in q_lower:
            target_item_keywords.extend(["研發費用", "research & development", "r&d"])
        elif "本期淨利" in query or "net income" in q_lower:
            target_item_keywords.extend(["本期淨利", "net income"])
        elif "eps" in q_lower or "每股盈餘" in query:
            target_item_keywords.extend(["每股盈餘", "eps"])
        elif "毛利" in query or "gross" in q_lower:
            target_item_keywords.extend(["毛利", "gross profit", "gross margin", "毛利率"])

        matched_vars = []
        if target_item_keywords:
            for k, v in extracted_vars.items():
                if any(kw in v["item"].lower() for kw in target_item_keywords):
                    matched_vars.append(v)

        if not matched_vars:
            matched_vars = list(extracted_vars.values())

        def find_same_item_pair(vars_list):
            grouped = {}
            for v in vars_list:
                key = re.sub(r'\W+', '_', v["item"]).strip('_').lower()
                grouped.setdefault(key, []).append(v)

            for key, item_vars in grouped.items():
                item_vars_sorted = sorted(item_vars, key=lambda x: str(x["year"]))
                if len(item_vars_sorted) >= 2 and item_vars_sorted[0]["year"] != item_vars_sorted[-1]["year"]:
                    return item_vars_sorted[0], item_vars_sorted[-1]

            sorted_all = sorted(vars_list, key=lambda x: str(x["year"]))
            if len(sorted_all) >= 2 and sorted_all[0]["year"] != sorted_all[-1]["year"]:
                return sorted_all[0], sorted_all[-1]
            return None, None

        if "cagr" in q_lower or "複合成長率" in q_lower:
            v1, v2 = find_same_item_pair(matched_vars)
            if not v1 and len(extracted_vars) >= 2:
                v1, v2 = find_same_item_pair(list(extracted_vars.values()))

            if v1 and v2:
                years = 1.0
                try:
                    years = abs(float(v2["year"]) - float(v1["year"]))
                    if years == 0:
                        years = 1.0
                except Exception:
                    years = 1.0
                item_label = v1['item']
                y1_label = v1['year']
                y2_label = v2['year']
                code_lines.append(f"cagr_val = cagr({v1['code_key']}, {v2['code_key']}, {years})")
                code_lines.append("result = round(cagr_val, 2)")
                code_lines.append(f"print(f'{item_label} CAGR ({y1_label} -> {y2_label}): {{result}}%')")
            else:
                code_lines.append("result = 0.0")

        elif "yoy" in q_lower or "成長率" in q_lower or "變動" in q_lower or "growth" in q_lower:
            v1, v2 = find_same_item_pair(matched_vars)
            if not v1 and len(extracted_vars) >= 2:
                v1, v2 = find_same_item_pair(list(extracted_vars.values()))

            if v1 and v2:
                item_label = v1['item']
                y1_label = v1['year']
                y2_label = v2['year']
                code_lines.append(f"growth_pct = yoy({v1['code_key']}, {v2['code_key']})")
                code_lines.append("result = round(growth_pct, 2)")
                code_lines.append(f"print(f'{item_label} YoY Growth ({y1_label} -> {y2_label}): {{result}}%')")
            else:
                code_lines.append("result = 0.0")

        elif "毛利率" in q_lower or "margin" in q_lower:
            profit_vars = []
            rev_vars = []
            for v in extracted_vars.values():
                item_lower = v["item"].lower()
                if "毛利" in item_lower or "gross" in item_lower:
                    profit_vars.append(v)
                if "收入" in item_lower or "revenue" in item_lower:
                    rev_vars.append(v)

            best_profit_var = None
            best_rev_var = None

            for p_var in profit_vars:
                for r_var in rev_vars:
                    if p_var["year"] == r_var["year"] and p_var["year"] != "N/A":
                        best_profit_var = p_var
                        best_rev_var = r_var
                        break
                if best_profit_var:
                    break

            if not best_profit_var and profit_vars and rev_vars:
                best_profit_var = profit_vars[0]
                best_rev_var = rev_vars[0]

            if best_profit_var and best_rev_var:
                p_year = best_profit_var['year']
                code_lines.append(f"margin_val = margin({best_profit_var['code_key']}, {best_rev_var['code_key']})")
                code_lines.append("result = round(margin_val, 2)")
                code_lines.append(f"print(f'Gross Margin ({p_year}): {{result}}%')")
            else:
                code_lines.append("result = 0.0")

        else:
            if matched_vars:
                first_v = matched_vars[0]
                item_label = first_v['item']
                y_label = first_v['year']
                code_lines.append(f"result = {first_v['code_key']}")
                code_lines.append(f"print(f'{item_label} ({y_label}): {{result}}')")
            elif extracted_vars:
                first_v = list(extracted_vars.values())[0]
                item_label = first_v['item']
                y_label = first_v['year']
                code_lines.append(f"result = {first_v['code_key']}")
                code_lines.append(f"print(f'{item_label} ({y_label}): {{result}}')")
            else:
                code_lines.append("result = 0.0")

        code_str = "\n".join(code_lines)

        # Sandbox Execution
        success, res_val, stdout_err = execute_pot_code(code_str)

        repair_count = 0
        while not success and repair_count < 2:
            repair_count += 1
            code_lines.append("result = 0.0")
            code_str = "\n".join(code_lines)
            success, res_val, stdout_err = execute_pot_code(code_str)

        extracted_summary = {v["code_key"]: (v["item"], v["year"], v["val"]) for v in extracted_vars.values()}

        return {
            "code": code_str,
            "success": success,
            "result_value": res_val,
            "output_log": stdout_err,
            "extracted_variables": extracted_summary,
            "repairs_triggered": repair_count
        }
