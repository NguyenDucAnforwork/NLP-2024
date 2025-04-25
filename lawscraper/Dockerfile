FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and cache the embedding model 
RUN mkdir -p /app/data/embeddings/bge-m3
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-m3'); model.save('/app/data/embeddings/bge-m3')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/crawled_data

# Environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "demo.py"]