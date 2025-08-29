# docker/Dockerfile
FROM python:3.11-slim

# Copy source code
COPY src /app

# Install needed packages
RUN pip install --no-cache-dir flask requests

# Expose container port
EXPOSE 9001

# Start the service
CMD ["python", "/app/proxy.py"]