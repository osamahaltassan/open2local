FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev linux-headers

# Install Python packages directly
RUN pip install --no-cache-dir --prefix=/install Flask==2.3.2 requests==2.31.0 gunicorn==21.2.0

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
    WHISPER_TIMEOUT=None

# Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9001", "--workers", "4", "--timeout", "0", "proxy:app"]