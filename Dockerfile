# Use a stable Python version
FROM python:3.10-slim

# Set user to root to ensure permissions for apt-get
USER root

# Modern replacement for libgl1-mesa-glx is libgl1 and libglx-mesa0
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies using the CPU-only index to save RAM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download MiDaS and timm during the build phase
RUN python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"

COPY . .

# Start the application using the PORT variable assigned by Render
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
