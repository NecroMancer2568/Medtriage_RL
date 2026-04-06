FROM python:3.11-slim

WORKDIR /app

# Install curl for health check in start.sh
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose both ports
# 7860 = FastAPI (OpenEnv spec endpoint, judges use this)
# 7861 = Gradio UI (interactive interface)
EXPOSE 7860 7861

# Launch both services
CMD ["./start.sh"]