# Base image with Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk stopwords
RUN python -m nltk.downloader stopwords

# Copy the rest of the app
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
