"""Naive dense retriever over ewha.pdf (baseline B1).

- Chunking: fixed-size character windows with overlap (no structure awareness).
- Embedding: Upstage solar-embedding-2 (passage model for chunks, query model for questions).
- Similarity: cosine (vectors are L2-normalized, so cosine == dot product).
- Embeddings are cached in index/ so the PDF is embedded only once.
"""
import hashlib
import json
import os
import re

import numpy as np
import pypdfium2 as pdfium

from llm import EMBED_PASSAGE_MODEL, EMBED_QUERY_MODEL, embed

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(HERE, "..", "ewha.pdf")  # shared course file in Project/
INDEX_DIR = os.path.join(HERE, "index")

# Lines repeated on every page (running header + page number like "2 - 2 - 6").
_HEADER_LINES = [
    re.compile(r"^\s*이화여자대학교\s*학칙\s*$"),
    re.compile(r"^\s*\d+\s*-\s*\d+\s*-\s*\d+\s*$"),
]


def load_pdf_text(pdf_path=DEFAULT_PDF):
    """Concatenate all pages, dropping running headers/page numbers."""
    pdf = pdfium.PdfDocument(pdf_path)
    lines = []
    for page in pdf:
        text = page.get_textpage().get_text_range()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if any(p.match(line) for p in _HEADER_LINES):
                continue
            lines.append(line)
    pdf.close()
    return "\n".join(lines)


def chunk_fixed(text, size=500, overlap=100):
    """Fixed-size character windows with overlap."""
    chunks, step = [], size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def _normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class NaiveRetriever:
    def __init__(self, pdf_path=DEFAULT_PDF, chunk_size=500, overlap=100):
        text = load_pdf_text(pdf_path)
        self.chunks = chunk_fixed(text, chunk_size, overlap)
        self.vectors = self._load_or_build(pdf_path, chunk_size, overlap)

    def _load_or_build(self, pdf_path, chunk_size, overlap):
        # Cache key changes whenever the PDF, chunking config, or model changes.
        with open(pdf_path, "rb") as f:
            pdf_hash = hashlib.md5(f.read()).hexdigest()[:8]
        key = f"naive_{pdf_hash}_{chunk_size}_{overlap}_{EMBED_PASSAGE_MODEL}"
        path = os.path.join(INDEX_DIR, key + ".npz")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=False)
            if json.loads(str(data["chunks"])) == self.chunks:
                return data["vectors"]
        vectors = _normalize(embed(self.chunks, EMBED_PASSAGE_MODEL))
        os.makedirs(INDEX_DIR, exist_ok=True)
        np.savez(path, vectors=vectors,
                 chunks=json.dumps(self.chunks, ensure_ascii=False))
        return vectors

    def search(self, query, k=5):
        """Top-k chunks by cosine similarity: [(score, chunk_id, text), ...]."""
        q = _normalize(embed([query], EMBED_QUERY_MODEL))[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), int(i), self.chunks[i]) for i in top]
