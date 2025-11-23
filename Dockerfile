FROM python:3.10-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy just the source code (includes pre-generated proto files in bbe_sdk/pb and bbe_sdk/proto)
COPY . /app/

# Set environment variables explicitly
# ENV DATASET_SERVICE_URL="http://localhost:8000/data"
# ENV CALIBRATION_SERVICE_URL="http://localhost:8001/probs"

# Default to MNIST, but can be overridden via docker-compose
CMD ["python", "main.py"]
