"""
RAG Store for NFZ Rule Snippets.

Stores and retrieves rule documentation snippets for LLM context.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class RuleSnippet(BaseModel):
    """NFZ rule documentation snippet."""

    snippet_id: str = Field(..., description="Unique snippet ID (e.g., CWV_001)")
    title: str = Field(..., description="Human-readable title")
    content: str = Field(..., description="Rule documentation text")
    source: str = Field(..., description="Source document name")
    url: str | None = Field(None, description="Source URL")
    rule_ids: list[str] = Field(default_factory=list, description="Related Priqualis rule IDs")


# =============================================================================
# RAG Store
# =============================================================================


class RAGStore:
    """
    Simple vector-free RAG store for rule snippets.
    
    Uses keyword matching for retrieval. For production,
    upgrade to proper vector embeddings + similarity search.
    """

    def __init__(self, snippets: list[RuleSnippet] | None = None):
        """
        Initialize store.

        Args:
            snippets: Pre-loaded snippets
        """
        self._snippets: dict[str, RuleSnippet] = {}
        if snippets:
            for s in snippets:
                self._snippets[s.snippet_id] = s

    @classmethod
    def from_yaml(cls, path: Path) -> "RAGStore":
        """
        Load snippets from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Initialized RAGStore
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Snippets file not found: %s", path)
            return cls()

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        snippets = []
        for item in data.get("snippets", []):
            snippets.append(RuleSnippet(
                snippet_id=item["id"],
                title=item.get("title", ""),
                content=item.get("content", ""),
                source=item.get("source", ""),
                url=item.get("url"),
                rule_ids=item.get("rule_ids", []),
            ))

        logger.info("Loaded %d snippets from %s", len(snippets), path)
        return cls(snippets)

    def add_snippet(self, snippet: RuleSnippet) -> None:
        """Add snippet to store."""
        self._snippets[snippet.snippet_id] = snippet

    def get_by_id(self, snippet_id: str) -> RuleSnippet | None:
        """Get snippet by ID."""
        return self._snippets.get(snippet_id)

    def get_by_rule(self, rule_id: str) -> list[RuleSnippet]:
        """Get all snippets related to a rule."""
        return [s for s in self._snippets.values() if rule_id in s.rule_ids]

    def search(self, query: str, top_k: int = 3) -> list[RuleSnippet]:
        """
        Search snippets by keyword matching.

        Args:
            query: Search query
            top_k: Maximum results to return

        Returns:
            List of matching snippets
        """
        query_lower = query.lower()
        scored = []

        for snippet in self._snippets.values():
            score = 0
            text = f"{snippet.title} {snippet.content}".lower()
            
            # Simple keyword matching
            for word in query_lower.split():
                if word in text:
                    score += 1

            if score > 0:
                scored.append((score, snippet))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [s for _, s in scored[:top_k]]

    def get_context(self, rule_id: str, max_chars: int = 1000) -> str:
        """
        Get context string for a rule.

        Args:
            rule_id: Rule identifier
            max_chars: Maximum context length

        Returns:
            Formatted context string
        """
        snippets = self.get_by_rule(rule_id)
        
        if not snippets:
            return f"No documentation found for rule {rule_id}."

        context_parts = []
        total_chars = 0

        for snippet in snippets:
            if total_chars + len(snippet.content) > max_chars:
                break
            
            context_parts.append(f"### {snippet.title}\n{snippet.content}\n*Source: {snippet.source}*")
            total_chars += len(snippet.content)

        return "\n\n".join(context_parts)

    @property
    def count(self) -> int:
        """Number of snippets in store."""
        return len(self._snippets)


# =============================================================================
# Default Snippets
# =============================================================================


DEFAULT_SNIPPETS = [
    RuleSnippet(
        snippet_id="CWV_001",
        title="Warunek CWV - Rozpoznanie główne",
        content=(
            "Świadczenie musi zawierać rozpoznanie główne w kodzie ICD-10. "
            "Rozpoznanie musi być zgodne z zakresem świadczeń określonym w umowie. "
            "Brak rozpoznania głównego powoduje odrzucenie świadczenia."
        ),
        source="NFZ CWV v2024",
        url="https://nfz.gov.pl/dla-swiadczeniodawcy/walidacje-i-weryfikacje/",
        rule_ids=["R001"],
    ),
    RuleSnippet(
        snippet_id="CWV_002",
        title="Warunek CWV - Daty hospitalizacji",
        content=(
            "Data wypisu musi być równa lub późniejsza niż data przyjęcia. "
            "Negatywny czas pobytu wskazuje na błąd w danych. "
            "Świadczenie z nieprawidłowym zakresem dat zostanie odrzucone."
        ),
        source="NFZ CWV v2024",
        rule_ids=["R002"],
    ),
    RuleSnippet(
        snippet_id="CWV_003",
        title="Warunek CWV - Kod JGP",
        content=(
            "Świadczenie szpitalne musi mieć przypisaną grupę JGP. "
            "Kod JGP określa sposób rozliczenia i wysokość taryfy. "
            "Brak kodu JGP uniemożliwia rozliczenie świadczenia."
        ),
        source="Opis algorytmu grupera JGP 2024",
        rule_ids=["R003"],
    ),
    RuleSnippet(
        snippet_id="CWV_010",
        title="Warunek CWV - Procedury medyczne",
        content=(
            "Świadczenie powinno zawierać co najmniej jedną procedurę medyczną. "
            "Procedury muszą być zgodne z grupą JGP. "
            "Brak procedur może skutkować weryfikacją lub odrzuceniem."
        ),
        source="NFZ CWV v2024",
        rule_ids=["R004"],
    ),
    RuleSnippet(
        snippet_id="CWV_015",
        title="Warunek CWV - Tryb przyjęcia",
        content=(
            "Tryb przyjęcia musi być określony jako: nagły, planowy lub przeniesienie. "
            "Nieprawidłowy tryb przyjęcia powoduje błąd walidacji."
        ),
        source="NFZ CWV v2024",
        rule_ids=["R005"],
    ),
    RuleSnippet(
        snippet_id="CWV_020",
        title="Warunek CWV - Kod oddziału",
        content=(
            "Świadczenie musi zawierać kod oddziału zgodny z rejestrem NFZ. "
            "Kod oddziału jest wymagany do prawidłowego rozliczenia."
        ),
        source="NFZ CWV v2024",
        rule_ids=["R006"],
    ),
    RuleSnippet(
        snippet_id="CWV_025",
        title="Warunek CWV - Wartość taryfy",
        content=(
            "Wartość taryfy musi być większa od zera. "
            "Świadczenie z zerową taryfą nie zostanie rozliczone."
        ),
        source="NFZ CWV v2024",
        rule_ids=["R007"],
    ),
]


def get_default_rag_store() -> RAGStore:
    """Get RAG store with default NFZ snippets."""
    return RAGStore(DEFAULT_SNIPPETS)
