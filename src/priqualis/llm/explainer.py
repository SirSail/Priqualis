"""
Violation Explainer using LLM.

Generates natural language explanations for validation results.
Uses RAG context from NFZ rule snippets.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from priqualis.rules.models import RuleResult
from priqualis.llm.rag import RAGStore, get_default_rag_store

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class Explanation(BaseModel):
    """LLM-generated explanation."""

    text: str = Field(..., description="Explanation text")
    citations: list[str] = Field(default_factory=list, description="Source citations")
    rule_id: str = Field(..., description="Related rule ID")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")


@dataclass(slots=True, frozen=True)
class ExplainerConfig:
    """
    Configuration for violation explainer.
    
    Attributes:
        model: LLM model name
        max_tokens: Maximum response tokens
        temperature: Sampling temperature
        language: Response language
    """

    model: str = "gpt-4o-mini"
    max_tokens: int = 300
    temperature: float = 0.3
    language: str = "pl"


DEFAULT_EXPLAINER_CONFIG = ExplainerConfig()


# =============================================================================
# Prompt Template
# =============================================================================


EXPLAIN_PROMPT_PL = """
Jesteś asystentem wyjaśniającym wyniki walidacji pakietów rozliczeniowych NFZ.

ZASADY:
- Wyjaśniaj TYLKO symboliczne wyniki - NIE podejmuj decyzji
- Cytuj źródła (podaj ID reguły i dokument)
- Używaj prostego języka zrozumiałego dla pracownika rozliczeń
- Odpowiedź max 3-4 zdania

KONTEKST Z BAZY REGUŁ:
{rag_context}

WYNIK WALIDACJI:
- Case ID: {case_id}
- Rule: {rule_id} - {rule_name}
- Status: {state}
- Message: {message}

Wyjaśnij dlaczego ten rekord otrzymał status {state} i co można zrobić.
"""

EXPLAIN_PROMPT_EN = """
You are an assistant explaining validation results for NFZ billing packages.

RULES:
- Explain ONLY symbolic results - DO NOT make decisions
- Cite sources (provide rule ID and document)
- Use simple language understandable by billing staff
- Response max 3-4 sentences

CONTEXT FROM RULE BASE:
{rag_context}

VALIDATION RESULT:
- Case ID: {case_id}
- Rule: {rule_id} - {rule_name}
- Status: {state}
- Message: {message}

Explain why this record received status {state} and what can be done.
"""


# =============================================================================
# LLM Client Protocol
# =============================================================================


class LLMClient(Protocol):
    """Protocol for LLM client."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        ...


# =============================================================================
# Violation Explainer
# =============================================================================


class ViolationExplainer:
    """
    Explains validation violations using LLM.
    
    Combines RAG context with LLM to generate
    human-readable explanations with citations.
    """

    def __init__(
        self,
        config: ExplainerConfig | None = None,
        rag_store: RAGStore | None = None,
        llm_client: LLMClient | None = None,
    ):
        """
        Initialize explainer.

        Args:
            config: Explainer configuration
            rag_store: RAG store for context
            llm_client: LLM client (optional, uses mock if None)
        """
        self.config = config or DEFAULT_EXPLAINER_CONFIG
        self.rag = rag_store or get_default_rag_store()
        self._client = llm_client

    def explain(self, violation: RuleResult, rule_name: str = "") -> Explanation:
        """
        Generate explanation for a violation.

        Args:
            violation: Violation result from rule engine
            rule_name: Human-readable rule name

        Returns:
            Explanation with text and citations
        """
        # Get RAG context
        context = self.rag.get_context(violation.rule_id)

        # Build prompt
        prompt = self._build_prompt(violation, rule_name, context)

        # Generate explanation
        if self._client:
            text = self._client.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        else:
            # Mock response when no LLM client
            text = self._generate_mock_explanation(violation, rule_name, context)

        # Extract citations
        citations = self._extract_citations(context)

        return Explanation(
            text=text,
            citations=citations,
            rule_id=violation.rule_id,
            confidence=1.0 if self._client else 0.8,
        )

    def explain_batch(
        self,
        violations: list[RuleResult],
        rule_names: dict[str, str] | None = None,
    ) -> list[Explanation]:
        """
        Generate explanations for multiple violations.

        Args:
            violations: List of violations
            rule_names: Mapping of rule_id to name

        Returns:
            List of explanations
        """
        rule_names = rule_names or {}
        return [
            self.explain(v, rule_names.get(v.rule_id, ""))
            for v in violations
        ]

    def _build_prompt(
        self,
        violation: RuleResult,
        rule_name: str,
        context: str,
    ) -> str:
        """Build prompt from template."""
        template = EXPLAIN_PROMPT_PL if self.config.language == "pl" else EXPLAIN_PROMPT_EN

        return template.format(
            rag_context=context,
            case_id=violation.case_id,
            rule_id=violation.rule_id,
            rule_name=rule_name or violation.rule_id,
            state=violation.state.value if hasattr(violation.state, 'value') else violation.state,
            message=violation.message or "No message",
        )

    def _generate_mock_explanation(
        self,
        violation: RuleResult,
        rule_name: str,
        context: str,
    ) -> str:
        """Generate mock explanation when no LLM available."""
        
        rule_explanations = {
            "R001": "Świadczenie wymaga rozpoznania głównego w kodzie ICD-10. Dodaj prawidłowy kod rozpoznania głównego zgodny z dokumentacją medyczną.",
            "R002": "Data wypisu musi być równa lub późniejsza niż data przyjęcia. Skoryguj daty hospitalizacji w systemie HIS.",
            "R003": "Każde świadczenie szpitalne wymaga przypisania grupy JGP. Uruchom grupera lub przypisz grupę ręcznie.",
            "R004": "Świadczenie powinno zawierać co najmniej jedną procedurę medyczną. Uzupełnij procedury zgodnie z dokumentacją.",
            "R005": "Tryb przyjęcia musi być określony jako: nagły, planowy lub przeniesienie. Wybierz prawidłową wartość.",
            "R006": "Kod oddziału jest wymagany do rozliczenia. Uzupełnij kod zgodny z rejestrem NFZ.",
            "R007": "Wartość taryfy musi być większa od zera. Sprawdź przypisanie grupy JGP i taryfy.",
        }

        base_text = rule_explanations.get(
            violation.rule_id,
            f"Reguła {violation.rule_id} nie została spełniona. {violation.message or ''}"
        )

        return f"{base_text} [Źródło: NFZ CWV v2024, reguła {violation.rule_id}]"

    def _extract_citations(self, context: str) -> list[str]:
        """Extract source citations from context."""
        citations = []
        
        # Simple extraction - look for "Source:" patterns
        for line in context.split("\n"):
            if "Source:" in line or "Źródło:" in line:
                citations.append(line.replace("*", "").strip())

        return citations


# =============================================================================
# Convenience Function
# =============================================================================


def explain_violation(
    violation: RuleResult,
    rule_name: str = "",
    config: ExplainerConfig | None = None,
) -> Explanation:
    """
    Convenience function to explain a single violation.

    Args:
        violation: Violation result
        rule_name: Human-readable rule name
        config: Explainer configuration

    Returns:
        Explanation
    """
    explainer = ViolationExplainer(config)
    return explainer.explain(violation, rule_name)
