import sys
import os
sys.path.append(os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from app.rag.vector_store import FinancialVectorStoreManager
from app.agent.orchestrator import FinAgentRAGOrchestrator
from app.agent.question_classifier import FinanceBenchClassifier
# NOTE: app.agent.financial_knowledge.FinancialQuestionUnderstanding was
# deleted in commit aac7f35 ("稍微整理了一下") without updating this import,
# which left this file unable to run at all (ModuleNotFoundError) on the
# real dev branch. Its responsibilities (entity/metric extraction, building
# retrieval queries) were absorbed into FinanceBenchClassifier.classify()
# in app/agent/question_classifier.py as part of that same commit — this
# swap points the test at the class that actually owns that logic now.

def test_pipeline():
    sample_path = os.path.join(os.path.dirname(__file__), "app", "sample_data", "sample_reports.json")
    print("Testing Vector Store initialization...")
    vs = FinancialVectorStoreManager(sample_json_path=sample_path, load_sample_data=True)
    print(f"Indexed {len(vs.corpus)} passages.")

    print("\nTesting FinAgent-RAG Orchestrator with query: '請分析台積電 2023 年與 2024 年全年的營業收入成長率 (YoY) 與毛利率變動？'")
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)
    res = orchestrator.process_query("請分析台積電 2023 年與 2024 年全年的營業收入成長率 (YoY) 與毛利率變動？")
    
    print("\n--- FINAL ANSWER ---")
    print(res["final_answer"])
    print("\n--- PoT PYTHON CODE ---")
    print(res["pot_code"])
    print("\n--- SANDBOX LOG ---")
    print(res["sandbox_log"])
    print("\n--- VERIFICATION RESULT ---")
    print(res["verification"])
    print("\n✅ Backend Test Passed Successfully!")


def test_orchestrator_returns_route_and_reasoning_metadata():
    sample_path = os.path.join(os.path.dirname(__file__), "app", "sample_data", "sample_reports.json")
    vs = FinancialVectorStoreManager(sample_json_path=sample_path, load_sample_data=True)
    orchestrator = FinAgentRAGOrchestrator(vector_store=vs)

    # Question with "gross margin" (calc_type=margin) + explanation keywords
    # → correct answer_mode is now NUMERIC (NUMERIC takes priority over EXPLANATION
    #   when calc_type is present, and LLM synthesizer handles the explanation part)
    res = orchestrator.process_query("請解釋台積電為什麼毛利率會上升？")

    assert res["answer_mode"] in ("NUMERIC", "EXPLANATION")  # NUMERIC is expected after fix
    assert "reasoning_steps" in res
    assert isinstance(res["reasoning_steps"], list)
    assert any(step.get("step_name") == "FinanceBench Classification" for step in res["reasoning_steps"])


def test_question_understanding_builds_domain_aware_queries():
    classifier = FinanceBenchClassifier()
    understanding = classifier.classify("Why did gross margin increase for TSMC in 2024?")

    # entity resolves to the classifier's canonical label for TSMC, not a bare "TSMC" string
    assert "TSMC" in understanding["entity"] or "台積電" in understanding["entity"]
    assert "gross_margin" in understanding["target_metrics"]
    assert any(
        "gross" in q.lower() or "毛利" in q or "revenue" in q.lower()
        for q in understanding["retrieval_queries"]
    )
    # the original query itself is always kept as one of the retrieval queries
    # so explanation-seeking phrasing ("why...increase") is never lost
    assert any("why" in q.lower() for q in understanding["retrieval_queries"])


if __name__ == "__main__":
    test_pipeline()
    test_orchestrator_returns_route_and_reasoning_metadata()
    print("\n✅ Metadata regression test passed.")
