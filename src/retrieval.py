"""
retrieval.py — Local semantic search with two-tier approach:

  Tier 1 (preferred): SentenceTransformers all-MiniLM-L6-v2
    - ~90MB download on first run, then fully offline
    - True semantic/meaning-based similarity

  Tier 2 (fallback): TF-IDF + cosine similarity (sklearn)
    - Zero download, 100% offline, works immediately
    - Keyword-frequency based similarity

The system automatically uses Tier 1 if the model is available,
and silently falls back to Tier 2 if not.
Both tiers use the same save/load/search interface.
"""

import json
import numpy as np
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger()

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
INDEX_FILE = DATA_DIR / "doc_index.json"
MODE_FILE = DATA_DIR / "index_mode.txt"    # records "transformer" or "tfidf"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Tier 1: SentenceTransformers
# ---------------------------------------------------------------------------

def _load_transformer_model():
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Tier 2: TF-IDF fallback (fully offline, zero download)
# ---------------------------------------------------------------------------

def _build_tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """Build TF-IDF vectors — works 100% offline with no model download."""
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    logger.info("Using TF-IDF fallback index (100% offline, no model download).")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    embeddings = normalize(matrix, norm="l2")

    # Persist vectorizer so query-time uses the same vocabulary
    vocab_file = DATA_DIR / "tfidf_vocab.pkl"
    with open(vocab_file, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"TF-IDF vocab saved → {vocab_file}")
    return embeddings


def _embed_query_tfidf(query: str) -> np.ndarray:
    import pickle
    from sklearn.preprocessing import normalize

    vocab_file = DATA_DIR / "tfidf_vocab.pkl"
    if not vocab_file.exists():
        raise FileNotFoundError("TF-IDF vocab not found. Run 'python main.py process' first.")
    with open(vocab_file, "rb") as f:
        vectorizer = pickle.load(f)
    vec = vectorizer.transform([query]).toarray().astype(np.float32)
    return normalize(vec, norm="l2")


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(docs: list[dict]) -> None:
    """
    Embed all documents and persist the index.

    Tries SentenceTransformers first; falls back to TF-IDF automatically.
    """
    valid_docs = [d for d in docs if d.get("text", "").strip()]
    if not valid_docs:
        logger.warning("No documents with text to index.")
        return

    texts = [d["text"][:512] for d in valid_docs]
    filenames = [d["filename"] for d in valid_docs]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mode = "tfidf"
    try:
        model = _load_transformer_model()
        logger.info(f"Embedding {len(texts)} document(s) with SentenceTransformers…")
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        mode = "transformer"
        logger.info("SentenceTransformer index built successfully.")
    except Exception as e:
        logger.warning(f"SentenceTransformer unavailable ({type(e).__name__}). Falling back to TF-IDF.")
        embeddings = _build_tfidf_embeddings(texts)

    np.save(str(EMBEDDINGS_FILE), embeddings)
    INDEX_FILE.write_text(json.dumps(filenames, indent=2), encoding="utf-8")
    MODE_FILE.write_text(mode, encoding="utf-8")
    logger.info(f"Search index saved [{mode} mode] → {EMBEDDINGS_FILE}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search documents by semantic similarity to the query.

    Returns:
        List of {filename, score} dicts sorted by descending score.
    """
    if not EMBEDDINGS_FILE.exists() or not INDEX_FILE.exists():
        raise FileNotFoundError(
            "No search index found. Run 'python main.py process' first."
        )

    embeddings = np.load(str(EMBEDDINGS_FILE))
    filenames: list[str] = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    mode = MODE_FILE.read_text(encoding="utf-8").strip() if MODE_FILE.exists() else "tfidf"

    if not filenames:
        return []

    # Embed query with same method used for the index
    if mode == "transformer":
        try:
            model = _load_transformer_model()
            query_vec = model.encode([query], normalize_embeddings=True)
        except Exception:
            logger.warning("Transformer unavailable for query; using TF-IDF fallback.")
            query_vec = _embed_query_tfidf(query)
    else:
        query_vec = _embed_query_tfidf(query)

    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    mode_label = "semantic (SentenceTransformer)" if mode == "transformer" else "TF-IDF (offline fallback)"
    logger.info(f"Search mode: {mode_label}")

    return [
        {"filename": filenames[i], "score": float(round(scores[i], 4))}
        for i in top_indices
        if scores[i] > 0.0
    ]
