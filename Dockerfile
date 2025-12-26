# Use a Python base image with lower overhead
FROM python:3.10-slim

# Install system dependencies for OpenCV and Torch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the MiDaS model during build time
# This saves time during startup and prevents re-downloads
RUN python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')"

COPY . .

# Set the start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
