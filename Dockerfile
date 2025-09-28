FROM python:3.11-slim

# Install system dependencies (added git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxrender1 libxext6 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
