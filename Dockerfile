# Lightweight Python container for the Voice-to-Do SLM
FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered terminal logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

# Install PyTorch CPU-optimized wheel to keep the Docker image small and fast
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy

# Copy model architecture, tokenizer, trained weights, and server files
COPY model.py tokenizer.py server.py vocab.json checkpoint_best.pt ./
COPY web/ ./web/

# Create persistent storage folder for tasks
RUN mkdir -p /app/data

# Expose web server port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

# Start the SLM inference & task server
CMD ["python", "server.py"]
