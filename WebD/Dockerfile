# ==============================================================================
# Base Image: Python 3.10 Slim
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for SPICE simulation and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ngspice \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Copy dependency definition and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Link weights to root/backend if present for model detector
RUN if [ -f /app/weights/epoch_40.pt ]; then \
        ln -sf /app/weights/epoch_40.pt /app/epoch_40.pt && \
        ln -sf /app/weights/epoch_40.pt /app/backend/epoch_40.pt; \
    fi && \
    if [ -f /app/weights/bjt_best_model.pt ]; then \
        ln -sf /app/weights/bjt_best_model.pt /app/bjt_best_model.pt && \
        ln -sf /app/weights/bjt_best_model.pt /app/backend/bjt_best_model.pt; \
    fi && \
    if [ -f /app/weights/best.pt ]; then \
        ln -sf /app/weights/best.pt /app/best.pt && \
        ln -sf /app/weights/best.pt /app/backend/best.pt; \
    fi

# Expose backend API port
EXPOSE 7860

# Launch FastAPI backend application using Uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
