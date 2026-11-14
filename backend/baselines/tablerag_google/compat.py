"""Adapter that lets Google's TableRAG code (google-research/table_rag, Apache-2.0) run on gpt-5-mini.

The third-party source is NOT vendored: point TABLERAG_SRC at the downloaded `table_rag` folder. Nothing in it is
edited; this module installs stand-in modules and patches `agent.model.Model` at import time. Every difference
from the original code is listed here so it can be reported in the thesis:

1. Imports of vertexai / transformers / langchain_google_vertexai (unused for OpenAI models) get stub modules.
2. langchain 1.x dropped `langchain.docstore.document` and `langchain.retrievers`; they are mapped to
   `langchain_core.documents.Document` and `langchain_classic.retrievers.EnsembleRetriever`.
3. `Model.get_context_limit` only knows old model names; gpt-5* get 400000. The tokenizer is o200k_base.
4. gpt-5 reasoning models reject `temperature`, `top_p`, `stop` and `max_tokens`. The patched query drops the
   first three, sends `max_completion_tokens` (reasoning tokens count against it, so the budget is larger than
   the original 128), and cuts the reply at the first stop string ("Observation:") itself, as the API would have.
5. Every chat call is written to the local usage ledger (`record_usage`), like every other call site.
"""
import os
import sys
import types

MAX_COMPLETION_TOKENS = int(os.environ.get("TABLERAG_MAX_COMPLETION_TOKENS", "4000"))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    class _Any:
        def __init__(self, *a, **k):
            raise RuntimeError("Vertex AI / HuggingFace are not available in this adapter (OpenAI models only)")

    _enum = type("_Enum", (), {"__getattr__": lambda self, n: n})()
    _stub("vertexai", init=lambda *a, **k: None)
    _stub("vertexai.preview")
    _stub("vertexai.preview.generative_models", GenerationConfig=_Any, GenerativeModel=_Any,
          HarmCategory=_enum, HarmBlockThreshold=_enum)
    _stub("transformers", AutoTokenizer=_Any)
    _stub("langchain_google_vertexai", VertexAIEmbeddings=_Any)
    from langchain_core.documents import Document
    from langchain_classic.retrievers import EnsembleRetriever
    _stub("langchain.docstore")
    _stub("langchain.docstore.document", Document=Document)
    _stub("langchain.retrievers", EnsembleRetriever=EnsembleRetriever)


def _patch_model(model_module):
    import tiktoken

    Model = model_module.Model
    orig_limit = Model.get_context_limit

    def get_context_limit(self, model_name):
        if model_name.startswith("gpt-5"):
            return 400000
        return orig_limit(self, model_name)

    def init(self, model_name):
        from openai import OpenAI
        self.model_name = model_name
        self.provider = "openai"
        self.context_limit = get_context_limit(self, model_name)
        self.client = OpenAI()
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def query_openai(self, prompt, system=None, rate_limit_per_minute=None, **kwargs):
        from app.agent import usage_tracker
        stop = kwargs.pop("stop", None) or []
        for unsupported in ("temperature", "top_p", "max_tokens"):
            kwargs.pop(unsupported, None)
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        response = self.query_openai_with_retry(messages, max_completion_tokens=MAX_COMPLETION_TOKENS, **kwargs)
        usage_tracker.record_usage(self.model_name, "tablerag_google", getattr(response, "usage", None), len(prompt))
        text = response.choices[0].message.content or ""
        for s in stop:
            cut = text.find(s)
            if cut >= 0:
                text = text[:cut]
        return text

    Model.get_context_limit = get_context_limit
    Model.__init__ = init
    Model.query_openai = query_openai


def load(src_dir=None):
    """Make Google's `agent`, `utils`, `prompts`, `evaluate` modules importable and patched; return the package."""
    src_dir = src_dir or os.environ.get("TABLERAG_SRC")
    if not src_dir or not os.path.isdir(src_dir):
        raise FileNotFoundError("set TABLERAG_SRC to the downloaded google-research/table_rag folder")
    backend = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend not in sys.path:
        sys.path.insert(0, backend)
    _install_stubs()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import agent.model as model_module
    _patch_model(model_module)
    import agent as agent_pkg
    import agent.retriever  # noqa: F401  (fails here if the langchain mapping is wrong)
    return agent_pkg
