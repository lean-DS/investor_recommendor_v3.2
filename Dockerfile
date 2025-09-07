FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    libopenblas-dev liblapack-dev libblas-dev \
    libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# ⬇️ Point to your real file: portfolio_recommender_v3_2_app.py
CMD ["bash", "-lc", "streamlit run portfolio_recommender_v3_2_app.py --server.address=0.0.0.0 --server.port=$PORT --browser.gatherUsageStats=false"]
