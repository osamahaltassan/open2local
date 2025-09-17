# docker/Dockerfile
FROM python:3.11-alpine AS builder

# Install dependencies into a temporary directory
RUN apk add --no-cache gcc musl-dev linux-headers
COPY /Docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --target=/install -r /tmp/requirements.txt

# Final stage
FROM python:3.11-alpine

# Copy only the installed packages
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Copy source code
COPY src /app

# Expose container port
EXPOSE 9001

# Set the endpoint for Whisper ASR Webservice API
ENV WHISPER_URL=http://localhost:9000

ENV DEBUG=true

# Start the service
CMD ["python", "/app/proxy.py"]