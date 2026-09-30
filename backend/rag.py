import json
import os

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(ROOT, "storage")
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
META_PATH = os.path.join(STORAGE_DIR, "chunks.json")

_embedder = None


def get_embedder() -> SentenceTransformer:
    # Loaded lazily so importing the module (e.g. for tests) stays fast
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return " ".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    # Overlapping windows so a sentence cut at a boundary still appears whole in one chunk
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_faiss_index(chunks: list):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    vectors = get_embedder().encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)
    # JSON instead of pickle: loading a pickle can execute arbitrary code
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)


def retrieve_chunks(query: str, top_k: int = 3):
    if not index_exists():
        raise FileNotFoundError("No index found. Upload a PDF via /ingest/upload first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    query_vec = get_embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, idx = index.search(query_vec.astype(np.float32), min(top_k, len(chunks)))
    return [chunks[i] for i in idx[0] if i >= 0]
