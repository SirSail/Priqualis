"""Tests for impact scoring."""

import pytest

from priqualis.rules.scoring import ImpactScorer, ScoringWeights, DEFAULT_WEIGHTS
from priqualis.rules.models import RuleResult, RuleState


class TestImpactScorer:
    """Tests for ImpactScorer."""

    @pytest.fixture
    def scorer(self) -> ImpactScorer:
        return ImpactScorer()

    def test_calculate_impact_violation(self, scorer: ImpactScorer, sample_claim_dict: dict):
        """Test impact calculation for violation."""
        result = RuleResult(
            rule_id="R001",
            case_id="ENC123",
            state=RuleState.VIOL,
            message="Error",
        )

        impact = scorer.calculate(result, sample_claim_dict)

        assert isinstance(impact, float)
        assert impact > 0

    def test_calculate_impact_sat_is_zero(self, scorer: ImpactScorer, sample_claim_dict: dict):
        """Test impact is 0 for SAT."""
        result = RuleResult(
            rule_id="R001",
            case_id="ENC123",
            state=RuleState.SAT,
        )

        impact = scorer.calculate(result, sample_claim_dict)

        assert impact == 0.0

    def test_impact_scales_with_tariff(self, scorer: ImpactScorer, sample_claim_dict: dict):
        """Test that higher tariff = higher impact."""
        result = RuleResult(
            rule_id="R001",
            case_id="ENC123",
            state=RuleState.VIOL,
            message="Error",
        )

        low_tariff = {**sample_claim_dict, "tariff_value": 1000.0}
        high_tariff = {**sample_claim_dict, "tariff_value": 10000.0}

        impact_low = scorer.calculate(result, low_tariff)
        impact_high = scorer.calculate(result, high_tariff)

        assert impact_high > impact_low


class TestScoringWeights:
    """Tests for ScoringWeights configuration."""

    def test_default_weights(self):
        """Test default weights are reasonable."""
        weights = ScoringWeights()

        assert weights.error_rejection_risk > 0
        assert weights.warning_rejection_risk > 0
        assert weights.tariff_baseline > 0

    def test_custom_weights(self):
        """Test custom weights can be set."""
        weights = ScoringWeights(
            error_rejection_risk=0.95,
            warning_rejection_risk=0.4,
            tariff_baseline=10000.0,
        )

        assert weights.error_rejection_risk == 0.95
        assert weights.warning_rejection_risk == 0.4
        assert weights.tariff_baseline == 10000.0

    def test_default_weights_singleton(self):
        """Test DEFAULT_WEIGHTS exists."""
        assert DEFAULT_WEIGHTS is not None
        assert isinstance(DEFAULT_WEIGHTS, ScoringWeights)
