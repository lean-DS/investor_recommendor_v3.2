FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    # Correct Streamlit env variable names:
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    libopenblas-dev liblapack-dev libblas-dev \
    libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependencies layer
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Code layer (rebuilt when GIT_SHA changes)
COPY . .

# This ARG is set from Cloud Build and forces the code layer to rebuild
ARG GIT_SHA=dev
LABEL git.sha=$GIT_SHA

# Sanity: show the first 40 lines of the exact file baked into the image
RUN echo "==== portfolio_recommender_v3_2_app.py (head) ====" && \
    sed -n '1,40p' portfolio_recommender_v3_2_app.py

EXPOSE 8080

CMD ["streamlit", "run", "portfolio_recommender_v3_2_app.py", "--server.address=0.0.0.0", "--server.port=8080"]
