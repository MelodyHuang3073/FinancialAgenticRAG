# Environment for ReAcTable (E1 in compat.py). Versions follow the repository's requirements.txt where it matters
# for behaviour; numpy and httpx are pinned only because the old pandas / openai releases need it.
FROM python:3.9-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential >/dev/null && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -q pybind11==2.10.4 "numpy<2" pandas==1.5.3 pandasql==0.7.3 SQLAlchemy==1.4.46 \
    openai==1.23.2 "httpx<0.28" python-dotenv==1.0.0 joblib==1.2.0 natsort==8.3.1 python-dateutil tqdm
RUN pip install --no-cache-dir -q pylcs==0.0.8
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
