FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for build tools)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose the port
EXPOSE 5000

# Start the application using Gunicorn
# Adjust workers and threads based on resource availability
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120 app:app
