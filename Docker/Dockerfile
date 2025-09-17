FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev linux-headers

# Copy requirements
COPY Docker/requirements.txt /build/requirements.txt
WORKDIR /build

# Install all dependencies to a directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final stage
FROM python:3.11-alpine

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy application source code
COPY src /app

# Create non-root user
RUN adduser -D proxyuser
USER proxyuser

# Expose port
EXPOSE 9001

# Variables
ENV WHISPER_URL=http://localhost:9000 \
    DEBUG=false \
    WHISPER_TIMEOUT=300

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "4", "proxy:app"]