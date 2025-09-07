# Dockerfile

# --- Base image ---
FROM python:3.11-slim

# --- Set workdir ---
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Copy requirements first (for caching layers) ---
COPY requirements.txt .

# --- Install Python deps ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy app code ---
COPY . .

# --- Expose Streamlit port ---
EXPOSE 8080

# --- Set Streamlit config for Cloud Run ---
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# --- Launch ---
CMD ["streamlit", "run", "app.py"]
