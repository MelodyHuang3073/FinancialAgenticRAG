"""Offline checks for the plain-RAG control (no API call, no key needed)."""
import json
import os
import pickle
import re

import pytest
from rank_bm25 import BM25Okapi

import run

HERE = os.path.dirname(os.path.abspath(__file__))


def test_retrieve_ranks_by_plain_bm25_and_takes_k():
    chunks = ["apples and pears", "net sales rose 5 percent", "sales sales sales net", "unrelated text here"] * 5
    bm = BM25Okapi([run.tokenize(c) for c in chunks])
    top = run.retrieve(bm, "Net Sales", k=3)
    scores = bm.get_scores(["net", "sales"])
    assert len(top) == 3 and len(set(top)) == 3
    assert [scores[j] for j in top] == sorted(scores, reverse=True)[:3]


def test_prompt_is_only_context_and_question():
    p = run.build_prompt(["chunk one", "chunk two"], "What was revenue?")
    assert p == "Context:\nchunk one\n\nchunk two\n\nQuestion: What was revenue?"


def test_answer_sends_one_user_message_and_nothing_else():
    seen = {}

    class Message:
        content = "ok"

    class Choice:
        message = Message

    class Usage:
        prompt_tokens, completion_tokens = 3, 4

    class Response:
        choices = [Choice]
        usage = Usage

    class Completions:
        @staticmethod
        def create(**kw):
            seen.update(kw)
            return Response

    class Chat:
        completions = Completions

    class Client:
        chat = Chat

    text, usage = run.answer(Client, "m", "PROMPT")
    assert text == "ok" and usage == (3, 4)
    assert seen == {"model": "m", "messages": [{"role": "user", "content": "PROMPT"}]}


def test_nothing_of_the_system_is_imported():
    for fn in ("run.py", "build_corpus.py"):
        src = open(os.path.join(HERE, fn), encoding="utf-8").read()
        assert not re.search(r"^\s*(from|import)\s+app\b", src, re.M)


@pytest.mark.skipif(not os.path.exists(os.path.join(run.TESTS, "original_clean_corpus_cache.pkl")),
                    reason="corpus not built")
def test_built_corpus_is_plain_text_only_and_covers_every_question():
    for qa, cache in run.SETS.values():
        corpus = pickle.load(open(os.path.join(run.TESTS, cache), "rb"))
        assert isinstance(corpus, dict)
        assert all(isinstance(c, str) and len(c) <= 1000 for v in corpus.values() for c in v)
        names = {q["doc_name"] for q in json.load(open(os.path.join(run.TESTS, qa), encoding="utf-8"))}
        assert all(run.find_doc(corpus, n) for n in names)


def test_find_doc_ignores_punctuation_and_the_word_dated():
    corpus = {"AMCOR_2022_8K_2022-07-01": [], "AMCOR_2023_10K": []}
    assert run.find_doc(corpus, "AMCOR_2022_8K_dated-2022-07-01") == "AMCOR_2022_8K_2022-07-01"
    assert run.find_doc(corpus, "AMCOR_2023_10K") == "AMCOR_2023_10K"
    with pytest.raises(SystemExit):
        run.find_doc(corpus, "AMCOR_2024_10K")
