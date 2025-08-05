# Base image
FROM python:3.12-slim

EXPOSE 5000

# Set system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Working directory \
WORKDIR /app

# Copy progect files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
\
# Set environment variables
ENV PYTHONUNBUFFERED=1

# Launch the application command
CMD ["python", "main.py"]