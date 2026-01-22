"""RAG Store for NFZ Rule Snippets."""

import logging
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RuleSnippet(BaseModel):
    snippet_id: str
    title: str
    content: str
    source: str
    url: str | None = None
    rule_ids: list[str] = Field(default_factory=list)

class RAGStore:
    """Simple keyword-based RAG store."""
    def __init__(self, snippets: list[RuleSnippet] | None = None):
        self._snippets = {s.snippet_id: s for s in snippets or []}

    @classmethod
    def from_yaml(cls, path: Path) -> "RAGStore":
        if not path.exists(): return cls()
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls([RuleSnippet(snippet_id=i["id"], title=i.get("title",""), content=i.get("content",""),
                               source=i.get("source",""), url=i.get("url"), rule_ids=i.get("rule_ids",[])) 
                   for i in data.get("snippets", [])])

    def get_by_rule(self, rule_id: str) -> list[RuleSnippet]:
        return [s for s in self._snippets.values() if rule_id in s.rule_ids]

    def get_context(self, rule_id: str, max_chars: int = 1000) -> str:
        snippets = self.get_by_rule(rule_id)
        if not snippets: return f"No documentation for {rule_id}."
        
        parts, total = [], 0
        for s in snippets:
            if total + len(s.content) > max_chars: break
            parts.append(f"### {s.title}\n{s.content}\n*Source: {s.source}*")
            total += len(s.content)
        return "\n\n".join(parts)

DEFAULT_SNIPPETS = [
    RuleSnippet(snippet_id="CWV_001", title="Rozpoznanie główne", content="Świadczenie musi zawierać rozpoznanie główne (ICD-10) zgodne z umową.", source="NFZ CWV v2024", rule_ids=["R001"]),
    RuleSnippet(snippet_id="CWV_002", title="Daty hospitalizacji", content="Data wypisu >= data przyjęcia.", source="NFZ CWV v2024", rule_ids=["R002"]),
    RuleSnippet(snippet_id="CWV_003", title="Kod JGP", content="Wymagany kod JGP dla świadczenia szpitalnego.", source="Algorytm JGP 2024", rule_ids=["R003"]),
    RuleSnippet(snippet_id="CWV_010", title="Procedury", content="Wymagana min. jedna procedura medyczna zgodna z JGP.", source="NFZ CWV v2024", rule_ids=["R004"]),
    RuleSnippet(snippet_id="CWV_015", title="Tryb przyjęcia", content="Wymagany tryb: nagły, planowy lub przeniesienie.", source="NFZ CWV v2024", rule_ids=["R005"]),
    RuleSnippet(snippet_id="CWV_020", title="Kod oddziału", content="Wymagany kod oddziału zgodny z rejestrem.", source="NFZ CWV v2024", rule_ids=["R006"]),
    RuleSnippet(snippet_id="CWV_025", title="Taryfa", content="Taryfa > 0.", source="NFZ CWV v2024", rule_ids=["R007"]),
]

def get_default_rag_store() -> RAGStore:
    return RAGStore(DEFAULT_SNIPPETS)
