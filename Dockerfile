
FROM python:3.11-slim

# Prevent Python from buffering logs and creating bytecode files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project into the container
COPY . .

# Run the stock prediction application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0","--port","8000"]
