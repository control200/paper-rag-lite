import io
from dataclasses import dataclass

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: int
    text: str


class SimpleRAG:
    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
        )
        self.chunk_matrix = None

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.chunk_matrix is not None

    def load_pdf_bytes(self, content: bytes) -> int:
        reader = PdfReader(io.BytesIO(content))

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(text)

        full_text = "\n".join(pages)

        if not full_text.strip():
            raise ValueError("No readable text was found in this PDF.")

        self.build_index(full_text)
        return len(self.chunks)

    def build_index(self, text: str) -> None:
        chunk_texts = self._split_text(text)

        self.chunks = [
            Chunk(chunk_id=i, text=chunk_text)
            for i, chunk_text in enumerate(chunk_texts)
        ]

        self.chunk_matrix = self.vectorizer.fit_transform(
            [chunk.text for chunk in self.chunks]
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.ready:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_matrix)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for index in ranked_indices:
            results.append(
                {
                    "chunk_id": self.chunks[index].chunk_id,
                    "text": self.chunks[index].text,
                    "score": float(scores[index]),
                }
            )

        return results

    def _split_text(self, text: str) -> list[str]:
        text = " ".join(text.split())

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])

            if end == len(text):
                break

            start = max(0, end - self.overlap)

        return chunks
