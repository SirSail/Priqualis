"""Rule Scoring for Priqualis."""

import logging
from dataclasses import dataclass
from typing import Any

from priqualis.rules.models import RuleResult

logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class ScoringWeights:
    """Immutable weights for impact score calculation (Risk * Tariff * FixCost)."""
    error_rejection_risk: float = 0.9
    warning_rejection_risk: float = 0.3
    autofix_available_cost: float = 0.2
    manual_fix_cost: float = 1.0
    tariff_baseline: float = 5000.0

DEFAULT_WEIGHTS = ScoringWeights()

class ImpactScorer:
    """Calculates impact scores for prioritization."""

    def __init__(self, weights: ScoringWeights | None = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def calculate(self, result: RuleResult, record: dict[str, Any]) -> float:
        if result.is_satisfied: return 0.0

        record_data = record.model_dump() if hasattr(record, "model_dump") else record
        
        # 1. Risk
        state = result.state if isinstance(result.state, str) else result.state.value
        risk = self.weights.error_rejection_risk if state == "VIOL" else (
               self.weights.warning_rejection_risk if state == "WARN" else 0.0)

        # 2. Tariff
        tariff = record_data.get("tariff_value", self.weights.tariff_baseline)
        tariff_factor = tariff / self.weights.tariff_baseline

        # 3. Fix Cost
        fix_cost = self.weights.autofix_available_cost if result.autofix_hint else self.weights.manual_fix_cost

        impact = risk * tariff_factor * fix_cost
        return round(impact, 4)

    def score_batch(self, results: list[RuleResult], records: dict[str, dict]) -> list[tuple[RuleResult, float]]:
        """Return results sorted by impact score descending."""
        scores = [(res, self.calculate(res, records.get(res.case_id, {}))) for res in results]
        return sorted(scores, key=lambda x: x[1], reverse=True)
