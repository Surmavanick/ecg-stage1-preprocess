# Minimal Dockerfile for Stage 1
FROM python:3.11-slim

# System deps (opencv needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
RUN mkdir -p /app/output

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
