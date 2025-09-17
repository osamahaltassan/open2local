# Docker/Dockerfile
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev linux-headers

# Set working directory
WORKDIR /build

# Copy requirements from the same directory (Docker folder)
COPY Docker/requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# Final stage
FROM python:3.11-alpine

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy application source code
COPY src /app

# Create non-root user
RUN adduser -D proxyuser
USER proxyuser

# Expose port
EXPOSE 9001

# Environment variables
ENV WHISPER_URL=http://localhost:9000 \
    DEBUG=false \
    WHISPER_TIMEOUT=300

# Run app
CMD ["python", "/app/proxy.py"]