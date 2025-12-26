# Use a stable Python version
FROM python:3.10-slim

# Set user to root to ensure permissions for apt-get
USER root

# Install system dependencies with a fix for Exit Code 100
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies using the CPU-only index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download MiDaS and timm during build phase
RUN python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"

COPY . .

# Render uses the $PORT environment variable
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}
