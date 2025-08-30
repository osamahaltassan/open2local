# docker/Dockerfile
FROM python:3.11-alpine

# Copy source code
COPY src /app

# Install needed packages
RUN pip install --no-cache-dir flask requests

# Expose container port
EXPOSE 9001

# Set the endpoint for Whisper ASR Webservice API
ENV WHISPER_URL=http://localhost:9000

# Start the service
CMD ["python", "/app/proxy.py"]