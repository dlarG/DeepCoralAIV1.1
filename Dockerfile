# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies (OpenCV + Torch deps)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
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

# Copy backend application code ONLY
COPY backend/ .

# Railway provides PORT dynamically
EXPOSE 8080

# Start app with gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app"]
