# VITAL Backend Dockerfile
FROM python:3.10-slim

# Prevent python from writing pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest
COPY requirements.txt /app/

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py /app/
COPY haarcascade_frontalface_default.xml /app/
COPY core/ /app/core/
COPY web/ /app/web/
COPY agents/ /app/agents/
COPY uploads/ /app/uploads/

# Expose backend service port
EXPOSE 5002

# Run API server
CMD ["python", "app.py"]
