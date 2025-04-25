# Vietnamese Legal Document Assistant

A system for retrieving and answering questions about Vietnamese legal documents using advanced retrieval techniques and large language models.

## Features

- Parent-child chunking for improved context retrieval
- Hybrid vector and BM25 search for better results
- Auto-merging retriever that combines related information
- Integration with Google's Gemini LLM
- Gradio interface for easy interaction

## System Architecture

The system is built with a modular architecture:

- **Crawlers**: Collect legal documents from various sources
- **Vector Indexes**: Store and index documents using embedding models
- **Retrievers**: Retrieve relevant information using hybrid search strategies
- **Generators**: Generate answers based on retrieved information

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional)
- Qdrant Cloud account (or local Qdrant instance)
- Google AI API key for Gemini

### Method 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vietnamese-legal-assistant.git
   cd vietnamese-legal-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the embedding model:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-m3'); model.save('./data/embeddings/bge-m3')"
   ```

4. Create and configure `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your-gemini-api-key
   QDRANT_API_KEY=your-qdrant-api-key
   QDRANT_URL=your-qdrant-url
   ```

5. Run the application:
   ```bash
   python demo.py
   ```

### Method 2: Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vietnamese-legal-assistant.git
   cd vietnamese-legal-assistant
   ```

2. Create and configure `.env` file with your API keys (same as above).

3. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:7860

## Usage

1. Start the application and wait for it to connect to the cloud index.
2. Enter your question in Vietnamese about legal documents.
3. Review the answer and the sources of information.

## Crawling New Documents

To crawl new documents:

```bash
# Using the Playwright crawler
python -m src.backend.crawlers.playwright_crawler

# Using the Scrapy-based crawler
scrapy crawl lawspider -O data/crawled_data/laws_$(date +%Y%m%d).json
```

## Indexing Documents

If you have new documents to index:

```python
from src.backend.vector_indexes import ParentVectorIndex, ChildVectorIndex, LegalDocumentProcessor

# Load documents
legal_texts, document_ids = LegalDocumentProcessor.load_from_json("data/crawled_data/your_file.json")
documents = LegalDocumentProcessor.prepare_documents(legal_texts, document_ids)

# Create indexes
parent_index = ParentVectorIndex()
child_index = ChildVectorIndex()

# Index documents
parent_idx, parent_nodes = parent_index.create_parent_index(documents)
child_idx, child_nodes = child_index.create_child_index(documents, parent_nodes)
```

## Development

### Project Structure

```
lawscraper/
├── data/                     # Data storage
│   ├── crawled_data/         # Raw crawled data
│   ├── embeddings/           # Embedding models
│   └── indexes/              # Index storage
├── scripts/                  # Utility scripts
├── src/                      # Source code
│   └── backend/              # Backend components
│       ├── crawlers/         # Web crawlers
│       ├── retrievers/       # Retrieval components
│       ├── vector_indexes/   # Vector indexing
│       └── generators/       # Answer generation
├── .env                      # Environment variables
├── demo.py                   # Main application
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
└── requirements.txt          # Python dependencies
```

## License

MIT

## Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) for the retrieval framework
- [Google Gemini](https://ai.google.dev/gemini-api) for the LLM capabilities
- [Qdrant](https://qdrant.tech/) for vector storage
- [Hugging Face](https://huggingface.co/) for embedding models