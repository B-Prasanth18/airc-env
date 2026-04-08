FROM python:3.10-slim

WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself so server.app and env imports work
RUN pip install --no-cache-dir -e .

# Expose HF Spaces port
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "-m", "server.app"]