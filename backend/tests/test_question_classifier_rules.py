"""FinanceBenchClassifier rules: change verbs, corpus-derived company name, and "how much
... USD" -> NUMERIC (app/agent/question_classifier.py).

The narrative-topic-query bridge (a hand-tuned per-question keyword list,
including a former entry for "expense as a percent of net sales" questions) was removed from the classifier and replaced
by decomposer.QueryDecomposer.suggest_narrative_topic_query(), a general
LLM-based suggester called from orchestrator.py -- not something
FinanceBenchClassifier.classify() does on its own anymore, so there is no
deterministic per-string assertion left to test here for that behavior.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agent.question_classifier import FinanceBenchClassifier


def test_classifier_change_verbs_and_foot_locker():
    clf = FinanceBenchClassifier()
    assert clf.classify("Did Acme grow its net PP&E from the prior year to the current year?")["calc_type"] == "change"
    assert clf.classify("Did cash holdings decline from the last year end to the second quarter?")["calc_type"] == "change"
    # a "which ... increase the most" selection question keeps no calc type
    assert clf.classify("Which segment posted the largest proportional sales rise?")["calc_type"] == ""
    # Company names are resolved against the uploaded corpus (no built-in list):
    # the readable name comes from the query's own spelling of the filename stem.
    from app.agent.orchestrator import FinAgentRAGOrchestrator
    assert FinAgentRAGOrchestrator._readable_company_name(
        "SPORTSDEPOT_2022_8K_dated_2022-08-19", "Does Sports Depot's new chief have prior top-executive experience?"
    ) == "Sports Depot"


def test_how_much_usd_question_is_numeric():
    c = FinanceBenchClassifier().classify(
        "How much does Acme expect to still pay for separating its unit, in USD million?"
    )
    assert c["answer_mode"] == "NUMERIC"
