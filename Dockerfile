# მსუბუქი Python base image
FROM python:3.11-slim

# Env config
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies (OpenCV და სხვა libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# სამუშაო დირექტორია
WORKDIR /app

# requirements.txt ქეშისთვის
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# app კოდი
COPY app ./app

# output დირექტორია
RUN mkdir -p /app/output

# uvicorn start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
