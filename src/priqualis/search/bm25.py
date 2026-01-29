"""
BM25 Search Module (Sparse Retrieval).
Wrapper for bm25s library handling tokenization and index management.
"""
import logging
from pathlib import Path
import bm25s
from priqualis.core.exceptions import SearchError

logger = logging.getLogger(__name__)

class SimpleTokenizer:
    def __init__(self, lowercase: bool = True): self.lowercase = lowercase
    def tokenize(self, text: str) -> list[str]: return (text.lower() if self.lowercase else text).split()
    def tokenize_batch(self, texts: list[str]) -> list[list[str]]: return [self.tokenize(t) for t in texts]

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: SimpleTokenizer | None = None):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.index = None
        self.corpus_ids = []
        self._is_built = False

    @property
    def is_built(self) -> bool: return self._is_built and self.index is not None

    def build(self, documents: list[tuple[str, str]]) -> None:
        if not documents: return
        self.corpus_ids = [doc[0] for doc in documents]
        texts = [doc[1] for doc in documents]
        
        corpus_tokens = bm25s.tokenize(texts, stopwords=None)
        logger.debug("Building BM25 k1=%.2f, b=%.2f", self.k1, self.b)
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        self._is_built = True

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        if not self.is_built: raise SearchError("BM25 index not built.")
        
        query_tokens = bm25s.tokenize([query], stopwords=None)
        assert self.index is not None
        results, scores = self.index.retrieve(query_tokens, k=min(top_k, len(self.corpus_ids)))
        
        return [(self.corpus_ids[idx], float(score)) for idx, score in zip(results[0], scores[0]) if idx < len(self.corpus_ids)]

    def save(self, path: Path) -> None:
        if not self.is_built: raise SearchError("Cannot save: index not built")
        path.mkdir(parents=True, exist_ok=True)
        assert self.index is not None
        self.index.save(str(path / "bm25_index"))
        
        import json
        (path / "corpus_ids.json").write_text(json.dumps(self.corpus_ids))
        logger.info("Saved BM25 to %s", path)

    def load(self, path: Path) -> None:
        if not path.exists(): raise SearchError(f"Path not found: {path}")
        self.index = bm25s.BM25.load(str(path / "bm25_index"), load_corpus=False)
        
        import json
        self.corpus_ids = json.loads((path / "corpus_ids.json").read_text())
        self._is_built = True
        logger.info("Loaded BM25 from %s (%d docs)", path, len(self.corpus_ids))

    def __len__(self) -> int: return len(self.corpus_ids)
