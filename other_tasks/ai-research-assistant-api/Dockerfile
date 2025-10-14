# -----------------------------
# Stage 1: Builder Stage
# -----------------------------
FROM python:3.12-slim AS builder

# Create the app directory
RUN mkdir /app

# Set working directory
WORKDIR /app

# Environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements.txt
COPY requirements.txt /app/

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Stage 2: Production Stage
# -----------------------------
FROM python:3.12-slim

# Create a non-root user
RUN useradd -m -r appuser && \
    mkdir /app && \
    chown -R appuser /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]
