"""Tests for BM25 search."""

import pytest

from priqualis.search.bm25 import BM25Index, SimpleTokenizer
from priqualis.core.exceptions import SearchError


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize("Hello World Test")

        assert tokens == ["hello", "world", "test"]

    def test_tokenize_no_lowercase(self):
        """Test tokenization without lowercasing."""
        tokenizer = SimpleTokenizer(lowercase=False)
        tokens = tokenizer.tokenize("Hello World")

        assert tokens == ["Hello", "World"]

    def test_tokenize_batch(self):
        """Test batch tokenization."""
        tokenizer = SimpleTokenizer()
        texts = ["Hello World", "Test Case"]
        tokens = tokenizer.tokenize_batch(texts)

        assert len(tokens) == 2
        assert tokens[0] == ["hello", "world"]
        assert tokens[1] == ["test", "case"]


class TestBM25Index:
    """Tests for BM25Index."""

    @pytest.fixture
    def index(self) -> BM25Index:
        return BM25Index(k1=1.5, b=0.75)

    @pytest.fixture
    def sample_documents(self) -> list[tuple[str, str]]:
        """Sample documents for indexing."""
        return [
            ("DOC1", "zapalenie płuc pneumonia JGP A01 procedura 88.761"),
            ("DOC2", "cukrzyca diabetes JGP B02 procedura 99.04"),
            ("DOC3", "zapalenie płuc ciężkie JGP A01 procedura 88.761 99.04"),
            ("DOC4", "nadciśnienie hypertension JGP C03"),
            ("DOC5", "zawał serca infarct JGP A02 procedura 37.22"),
        ]

    def test_build_index(self, index: BM25Index, sample_documents):
        """Test building index."""
        index.build(sample_documents)

        assert index.is_built
        assert len(index.corpus_ids) == 5
        assert len(index) == 5

    def test_search_returns_results(self, index: BM25Index, sample_documents):
        """Test search returns relevant results."""
        index.build(sample_documents)

        results = index.search("zapalenie płuc", top_k=3)

        assert len(results) <= 3
        # DOC1 or DOC3 should be in results (contain "zapalenie płuc")
        result_ids = [r[0] for r in results]
        assert "DOC1" in result_ids or "DOC3" in result_ids

    def test_search_empty_query(self, index: BM25Index, sample_documents):
        """Test search with empty query."""
        index.build(sample_documents)

        results = index.search("", top_k=3)

        # Empty query may return empty or zero-score results
        assert isinstance(results, list)

    def test_search_before_build_raises(self, index: BM25Index):
        """Test search before building raises error."""
        with pytest.raises(SearchError):
            index.search("query", top_k=3)

    def test_top_k_limit(self, index: BM25Index, sample_documents):
        """Test top_k limits results."""
        index.build(sample_documents)

        results = index.search("JGP", top_k=2)

        assert len(results) <= 2

    def test_search_returns_scores(self, index: BM25Index, sample_documents):
        """Test search returns (id, score) tuples."""
        index.build(sample_documents)

        results = index.search("zapalenie", top_k=3)

        for case_id, score in results:
            assert isinstance(case_id, str)
            assert isinstance(score, float)

    def test_build_empty_documents(self, index: BM25Index):
        """Test building with no documents."""
        index.build([])

        assert not index.is_built

    def test_save_before_build_raises(self, index: BM25Index, tmp_path):
        """Test saving before build raises error."""
        with pytest.raises(SearchError):
            index.save(tmp_path / "bm25")
