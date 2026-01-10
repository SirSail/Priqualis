"""Tests for hybrid search."""

import pytest
from unittest.mock import Mock, MagicMock

from priqualis.search.hybrid import HybridSearch
from priqualis.search.models import SearchQuery


class TestHybridSearch:
    """Tests for HybridSearch."""

    @pytest.fixture
    def mock_bm25(self):
        """Mock BM25 index."""
        mock = Mock()
        mock.is_built = True
        mock.search.return_value = [
            ("DOC1", 2.5),
            ("DOC2", 2.0),
            ("DOC3", 1.5),
        ]
        return mock

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        mock = Mock()
        # Qdrant returns ScoredPoint objects
        point1 = Mock(id="DOC2", score=0.95)
        point2 = Mock(id="DOC4", score=0.90)
        point3 = Mock(id="DOC1", score=0.85)
        mock.search.return_value = [point1, point2, point3]
        return mock

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        mock = Mock()
        mock.embed.return_value = [[0.1] * 384]
        return mock

    @pytest.fixture
    def hybrid_search(self, mock_bm25, mock_vector_store, mock_embedding_service):
        """HybridSearch with mocked dependencies."""
        return HybridSearch(
            bm25_index=mock_bm25,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            alpha=0.5,
        )

    def test_search_query_creation(self):
        """Test SearchQuery model."""
        query = SearchQuery(
            case_id="TEST",
            text="zapalenie płuc",
            jgp_code="A01",
            icd10_codes=["J18.9"],
            procedures=["88.761"],
        )

        assert query.case_id == "TEST"
        assert query.jgp_code == "A01"

    def test_alpha_property(self, hybrid_search):
        """Test alpha weighting parameter."""
        assert hybrid_search.alpha == 0.5

    def test_different_alpha_values(self, mock_bm25, mock_vector_store, mock_embedding_service):
        """Test different alpha values create different configurations."""
        high_bm25 = HybridSearch(
            bm25_index=mock_bm25,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            alpha=0.9,
        )

        low_bm25 = HybridSearch(
            bm25_index=mock_bm25,
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            alpha=0.1,
        )

        assert high_bm25.alpha != low_bm25.alpha
        assert high_bm25.alpha > low_bm25.alpha
