"""Violation Explainer using LLM."""

import logging
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field
from priqualis.llm.rag import RAGStore, get_default_rag_store
from priqualis.rules.models import RuleResult

logger = logging.getLogger(__name__)

class Explanation(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    rule_id: str
    confidence: float = 1.0

@dataclass(slots=True, frozen=True)
class ExplainerConfig:
    model: str = "gpt-4o-mini"; max_tokens: int = 300
    temperature: float = 0.3; language: str = "pl"

DEFAULT_EXPLAINER_CONFIG = ExplainerConfig()

EXPLAIN_PROMPT_PL = """Jesteś asystentem wyjaśniającym wyniki walidacji NFZ.
ZASADY: Wyjaśniaj tylko symboliczne wyniki, cytuj źródła, używaj prostego języka (max 3-4 zdania). Dodaj tłumaczenie EN po '🇬🇧 English:'.
KONTEKST: {rag_context}
WYNIK: {case_id} | {rule_id} - {rule_name} | {state} | {message}
Wyjaśnij status {state}."""

EXPLAIN_PROMPT_EN = """Assistant explaining NFZ validation results.
RULES: Symbolic explanation only, cite sources, simple language (3-4 sentences). Add PL translation after '🇵🇱 Polish:'.
CONTEXT: {rag_context}
RESULT: {case_id} | {rule_id} - {rule_name} | {state} | {message}
Explain status {state}."""

class ViolationExplainer:
    def __init__(self, config: ExplainerConfig | None = None, rag_store: RAGStore | None = None, llm_client: Any | None = None):
        self.config = config or DEFAULT_EXPLAINER_CONFIG
        self.rag = rag_store or get_default_rag_store()
        self._client = llm_client

    def explain(self, violation: RuleResult, rule_name: str = "") -> Explanation:
        ctx = self.rag.get_context(violation.rule_id)
        prompt = (EXPLAIN_PROMPT_PL if self.config.language == "pl" else EXPLAIN_PROMPT_EN).format(
            rag_context=ctx, case_id=violation.case_id, rule_id=violation.rule_id, 
            rule_name=rule_name or violation.rule_id, state=violation.state, message=violation.message or ""
        )
        
        txt = self._client.generate(prompt, max_tokens=self.config.max_tokens, temperature=self.config.temperature) if self._client else self._mock(violation)
        return Explanation(text=txt, citations=[l.replace("*", "").strip() for l in ctx.split("\n") if "Source:" in l], rule_id=violation.rule_id, confidence=1.0 if self._client else 0.8)

    def _mock(self, v: RuleResult) -> str:
        msgs = {
            "R001": ("Brak rozpoznania głównego (ICD-10).", "Missing main diagnosis."),
            "R002": ("Data wypisu < data przyjęcia.", "Discharge date error."),
            "R003": ("Brak grupy JGP.", "Missing JGP group."),
            "R005": ("Błędny tryb przyjęcia.", "Invalid admission mode.")
        }
        pl, en = msgs.get(v.rule_id, (f"Niespełniona reguła {v.rule_id}.", f"Rule {v.rule_id} failed."))
        return f"{pl} {v.message or ''}\n\n🇬🇧 English: {en}\n\n[Źródło: NFZ CWV v2024]"
