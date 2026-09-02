import sys, os
sys.path.insert(0, os.path.abspath("."))

from app.rag.vector_store import FinancialVectorStoreManager
from app.rag.parser import FinancialFileParser
from app.agent.orchestrator import FinAgentRAGOrchestrator

FIXTURES_DIR = os.path.join("tests", "financebench_pdfs")

QUERIES = [
    ("AMAZON_2017_10K.pdf", "What is Amazon's FY2017 days payable outstanding (DPO)? DPO is defined as: 365 * (average accounts payable between FY2016 and FY2017) / (FY2017 COGS + change in inventory between FY2016 and FY2017). Round your answer to two decimal places. Address the question by using the line items and information shown within the balance sheet and the P&L statement."),
    ("AMAZON_2017_10K.pdf", "What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income."),
    ("AMD_2015_10K.pdf", "Answer the following question as if you are an equity research analyst and have lost internet connection so you do not have access to financial metric providers. According to the details clearly outlined within the P&L statement and the statement of cash flows, what is the FY2015 depreciation and amortization (D&A from cash flow statement) % margin for AMD?"),
    ("AMERICANWATERWORKS_2021_10K.pdf", "Basing your judgments off of the cash flow statement and the income statement, what is American Water Works's FY2021 unadjusted operating income + depreciation and amortization from the cash flow statement (unadjusted EBITDA) in USD millions?"),
]

parser = FinancialFileParser()
_cache = {}

def get_store(filename):
    if filename in _cache:
        return _cache[filename]
    vs = FinancialVectorStoreManager()
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "rb") as f:
        content = f.read()
    result = parser.parse_file(filename, content)
    c_name = os.path.splitext(filename)[0]
    vs.add_parsed_passages(filename, c_name, result["passages"])
    _cache[filename] = vs
    return vs

for filename, q in QUERIES:
    print("=" * 100)
    print("FILE:", filename)
    print("Q:", q)
    vs = get_store(filename)
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)
    res = orchestrator.process_query(q)
    print("POT_CODE:")
    print(res.get("pot_code", ""))
    print("SANDBOX_LOG:")
    print(res.get("sandbox_log", ""))
    print("FINAL:", res.get("final_answer", ""))
    print()
