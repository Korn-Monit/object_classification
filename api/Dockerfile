FROM python:3.9-slim

WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home appuser

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application code, not the whole project
COPY --chown=appuser:appuser api/app /app/app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Set the GCS bucket and model blob name as environment variables
# These should be configured in your Cloud Run service settings
# ENV GCS_BUCKET_NAME="your-gcs-bucket-name"
# ENV MODEL_BLOB_NAME="path/to/your/model.pth"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

