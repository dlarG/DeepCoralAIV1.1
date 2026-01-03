# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies (OpenCV + Torch)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app
COPY backend/ .

# Railway injects PORT dynamically
EXPOSE 8080

# Run app
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app"]
