FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system dependencies (add as needed)
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Healthcheck (adjust endpoint if needed)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
