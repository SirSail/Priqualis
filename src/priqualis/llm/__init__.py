"""
Priqualis LLM Module.

Provides natural language explanations for validation results.
Uses RAG with NFZ rule snippets for context.
"""

from .explainer import (
    ViolationExplainer,
    ExplainerConfig,
    Explanation,
)
from .rag import (
    RAGStore,
    RuleSnippet,
)

__all__ = [
    "ViolationExplainer",
    "ExplainerConfig",
    "Explanation",
    "RAGStore",
    "RuleSnippet",
]
