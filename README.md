# PaperRAG Lite

A lightweight PDF question-answering application built with **FastAPI**, **TF-IDF retrieval**, and **Qwen**.

The project supports uploading a PDF document, automatically extracting and splitting its text, retrieving the most relevant content for a user question, and generating an answer based on the retrieved context.

## Features

- PDF upload and text extraction
- Automatic text chunking with overlap
- TF-IDF vectorization
- Cosine-similarity retrieval
- Top-K relevant chunk selection
- Qwen-based answer generation
- FastAPI REST API
- Swagger API documentation
- Demo mode without an API key

## Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
TF-IDF Vectorization
    ↓
Knowledge Index

User Question
    ↓
Query Vectorization
    ↓
Cosine Similarity
    ↓
Top-K Retrieval
    ↓
Retrieved Context
    ↓
Qwen
    ↓
Answer
```

## Project Structure

```text
paper-rag-lite/
├── app.py              # FastAPI application and API endpoints
├── rag.py              # PDF parsing, chunking and retrieval
├── llm.py              # Qwen API integration
├── test_rag.py         # Retrieval test
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable example
├── .gitignore
└── README.md
```

## Tech Stack

- Python 3.11
- FastAPI
- Pydantic
- PyPDF
- scikit-learn
- TF-IDF
- Cosine Similarity
- Qwen API
- Uvicorn

## Installation

Clone the repository and create a Python environment:

```bash
conda create -n paper-rag python=3.11 -y
conda activate paper-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy the example environment file:

### Windows

```bash
copy .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Edit `.env`:

```env
DASHSCOPE_API_KEY=your_api_key
QWEN_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

If `DASHSCOPE_API_KEY` is not configured, the application still runs in demo mode and returns the retrieved document context instead of calling Qwen.

## Run

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

FastAPI will provide an interactive Swagger interface for testing the APIs.

## API

### Health Check

```http
GET /health
```

Returns the current service status and whether a document has been loaded.

---

### Upload PDF

```http
POST /upload
```

Upload a PDF file.

Example response:

```json
{
  "filename": "paper.pdf",
  "chunks": 18,
  "message": "PDF indexed successfully."
}
```

---

### Ask a Question

```http
POST /ask
```

Request body:

```json
{
  "question": "What problem does this paper solve?",
  "top_k": 3
}
```

Example response:

```json
{
  "question": "What problem does this paper solve?",
  "answer": "The paper focuses on ...",
  "sources": [
    {
      "chunk_id": 2,
      "score": 0.7312
    }
  ]
}
```

## Retrieval Process

After a PDF is uploaded, the document text is divided into overlapping chunks.

Each chunk is converted into a TF-IDF vector:

```text
Document Chunk
      ↓
TF-IDF Vector
```

When a user asks a question, the query is converted using the same vectorizer:

```text
User Question
      ↓
Query Vector
```

Cosine similarity is then calculated between the query vector and all document chunk vectors.

The most relevant chunks are selected:

```text
Query
  ├── Chunk 1 → 0.18
  ├── Chunk 2 → 0.73
  ├── Chunk 3 → 0.41
  └── Chunk 4 → 0.82

Top-K
  ↓
Chunk 4
Chunk 2
Chunk 3
```

The retrieved text is combined with the user question and sent to Qwen to generate the final answer.

## Example

After uploading a research paper, you can ask questions such as:

```text
What is the main contribution of this paper?
```

```text
What dataset was used in the experiment?
```

```text
What are the limitations of the proposed method?
```

```text
Summarize the methodology described in the paper.
```

## Notes

This version uses **TF-IDF** for lightweight local retrieval and does not require a separate embedding model or vector database.

The retrieval module can later be extended with:

- Qwen Embedding
- BGE Embedding
- FAISS
- Qdrant
- Milvus
- Reranker models

## License

MIT
