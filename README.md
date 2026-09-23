# PaperRAG Lite

A minimal AI application project for learning and interviews.

**One task only: upload one PDF, ask a question, retrieve relevant chunks, and let Qwen answer from those chunks.**

## Why this project exists

The goal is not to build a large Agent platform. The goal is to understand one complete AI application flow:

```text
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
TF-IDF vectors
 ↓
Cosine similarity
 ↓
Top-K chunks
 ↓
Qwen
 ↓
Answer
```

## Tech stack

- Python
- FastAPI
- Pydantic
- PyPDF
- TF-IDF
- Cosine similarity
- Qwen API through DashScope's OpenAI-compatible endpoint

## Project structure

```text
paper-rag-lite/
├── app.py          # FastAPI API: /upload and /ask
├── rag.py          # PDF parsing, chunking and retrieval
├── llm.py          # Qwen API call
├── test_rag.py     # Minimal retrieval test
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

There are only three Python files you need to understand.

## 1. Install

```bash
conda create -n paper-rag python=3.11 -y
conda activate paper-rag
pip install -r requirements.txt
```

## 2. Configure Qwen

Copy:

```bash
copy .env.example .env
```

Then put your key in `.env`:

```env
DASHSCOPE_API_KEY=your_key_here
QWEN_MODEL=qwen-plus
```

If no key is configured, the project still runs in demo mode and shows the retrieved context instead of calling Qwen.

## 3. Run

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 4. Use

### Upload PDF

Use Swagger:

```text
POST /upload
```

Choose a PDF.

### Ask

```text
POST /ask
```

Example:

```json
{
  "question": "What problem does this paper solve?",
  "top_k": 3
}
```

## API flow

```text
POST /upload
    ↓
UploadFile
    ↓
PdfReader
    ↓
split_text()
    ↓
TfidfVectorizer.fit_transform()
    ↓
Knowledge index ready


POST /ask
    ↓
question
    ↓
TfidfVectorizer.transform()
    ↓
cosine_similarity()
    ↓
Top-K chunks
    ↓
Qwen API
    ↓
answer
```

## What to say in an interview

### What is RAG?

RAG means Retrieval-Augmented Generation. Instead of asking the LLM to answer only from its pretrained knowledge, the application first retrieves relevant information from an external document and then passes the retrieved context to the LLM.

### Does this project use an embedding model?

No. This learning version uses **TF-IDF** to convert text into vectors. It is deliberately simple and easy to understand. In a production version, TF-IDF can be replaced by a semantic embedding model such as Qwen/BGE embeddings, plus FAISS or a vector database.

### Why chunk the PDF?

A long PDF cannot be efficiently supplied to the model or retrieved as one unit. Chunking provides smaller retrieval units.

### Why overlap?

Overlap reduces information loss when a sentence or concept crosses a chunk boundary.

### Why cosine similarity?

It measures how similar the query vector direction is to each chunk vector. Higher similarity means the chunk is more relevant to the query.

### Why FastAPI?

FastAPI exposes the Python RAG logic as HTTP APIs so a web page, mobile app, or another backend can call it.

## Resume wording

Use only after you can explain the code yourself:

**PaperRAG Lite — PDF Knowledge Question Answering System**

- Built a lightweight document question-answering backend with Python, FastAPI and Qwen API.
- Implemented PDF parsing, text chunking, TF-IDF vectorization, cosine-similarity Top-K retrieval and context-augmented generation.
- Exposed document upload and question-answering functions through REST APIs with Pydantic validation.
- Designed the retrieval module so TF-IDF can later be replaced by semantic embeddings and a vector database.

## What is intentionally NOT included

This version does not include:

- Agent
- Function Calling
- MCP
- SQLite
- Redis
- SSE
- Docker
- vector database
- reranker

Those are useful later, but they are not required to understand the core AI application flow.
