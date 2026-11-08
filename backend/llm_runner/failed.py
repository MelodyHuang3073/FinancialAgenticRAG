"""Shared definition of a failed record in a model-swap run."""
FALLBACK = ("According to the financial report disclosures", "The calculation result for",
            "Analysis based on retrieved financial evidence", "Based on retrieved financial data for",
            "could not be obtained")


def failure(record):
    """Why a record does not count (None when it is a real model answer)."""
    ans = record.get("model_answer") or ""
    if record.get("error"):
        return "error " + str(record["error"])[:60]
    if not ans.strip():
        return "empty answer"
    if any(f in ans[:200] for f in FALLBACK):
        return "fallback-shaped answer (a model call failed and the system synthesised the text)"
    return None
