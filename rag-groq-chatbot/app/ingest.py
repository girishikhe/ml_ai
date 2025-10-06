import os
from typing import List
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from .retriever import VectorDB


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    sections = []
    start = 0
    while start < len(words):
        section = words[start:start + chunk_size]
        sections.append(" ".join(section))
        start += chunk_size - overlap
    return sections


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    content = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
            if txt:
                content.append(txt)
        except Exception:
            continue
    return "\n".join(content)


class PDFIngestor:
    def __init__(self, model_name: str, vector_db: VectorDB):
        self.encoder = SentenceTransformer(model_name)
        self.store = vector_db

    def process_pdf(self, file_path: str, doc_name: str = None) -> int:
        raw_text = read_pdf(file_path)
        parts = split_text(raw_text)

        # create embeddings
        vectors = self.encoder.encode(parts, show_progress_bar=False, convert_to_numpy=True)

        # store metadata
        meta = [{"document": doc_name or os.path.basename(file_path), "chunk_id": i, "content": parts[i]}
                for i in range(len(parts))]

        self.store.add(vectors, meta)
        return len(parts)
