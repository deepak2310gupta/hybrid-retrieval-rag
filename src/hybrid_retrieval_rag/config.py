from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DIRECTORY_PATH = BASE_DIR / "data"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DENSE_RETRIEVAL_TOP_K = 6
SPARSE_RETRIEVAL_TOP_K = 6
