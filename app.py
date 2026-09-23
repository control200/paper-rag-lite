import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from llm import ask_qwen
from rag import SimpleRAG

load_dotenv()

app = FastAPI(
    title="PaperRAG Lite",
    version="1.0.0",
    description="A minimal PDF question-answering project with FastAPI, RAG and Qwen.",
)

rag = SimpleRAG()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8)


@app.get("/")
def home():
    return {
        "name": "PaperRAG Lite",
        "message": "Upload a PDF at /upload, then ask questions at /ask.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": rag.ready,
        "chunks": len(rag.chunks),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        chunk_count = rag.load_pdf_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    return {
        "filename": filename,
        "chunks": chunk_count,
        "message": "PDF indexed successfully.",
    }


@app.post("/ask")
def ask(request: AskRequest):
    if not rag.ready:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF before asking questions.",
        )

    results = rag.search(request.question, top_k=request.top_k)

    context = "\n\n".join(
        f"[Chunk {item['chunk_id']}]\n{item['text']}"
        for item in results
    )

    answer = ask_qwen(
        question=request.question,
        context=context,
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "chunk_id": item["chunk_id"],
                "score": round(item["score"], 4),
            }
            for item in results
        ],
    }
