"""Tests for rule engine."""

from pathlib import Path

import pytest

from priqualis.rules.engine import RuleEngine, RuleExecutor, load_rules, safe_eval
from priqualis.rules.models import RuleDefinition, RuleResult, RuleState
from priqualis.etl.schemas import ClaimRecord


class TestLoadRules:
    """Tests for rule loading."""

    def test_load_rules_from_directory(self):
        """Test loading rules from config/rules directory."""
        rules_path = Path("config/rules")
        if not rules_path.exists():
            pytest.skip("config/rules not found")

        rules = load_rules(rules_path)

        assert len(rules) >= 1
        assert all(isinstance(r, RuleDefinition) for r in rules)

    def test_load_rules_from_sample(self, sample_rules_dir: Path):
        """Test loading rules from sample directory."""
        rules = load_rules(sample_rules_dir)

        assert len(rules) == 1
        assert rules[0].rule_id == "R001"


class TestSafeEval:
    """Tests for safe expression evaluation."""

    def test_safe_eval_basic(self):
        """Test basic expression evaluation."""
        context = {"x": 10, "y": 5}
        result = safe_eval("x > y", context)

        assert result is True

    def test_safe_eval_with_none(self):
        """Test evaluation with None values."""
        context = {"value": None}
        result = safe_eval("value is None", context)

        assert result is True

    def test_safe_eval_len_function(self):
        """Test len() is available."""
        context = {"items": [1, 2, 3]}
        result = safe_eval("len(items) > 0", context)

        assert result is True

    def test_safe_eval_str_function(self):
        """Test str() is available."""
        context = {"num": 42}
        result = safe_eval("str(num) == '42'", context)

        assert result is True


class TestRuleExecutor:
    """Tests for RuleExecutor."""

    @pytest.fixture
    def executor(self) -> RuleExecutor:
        return RuleExecutor()

    @pytest.fixture
    def sample_rule(self) -> RuleDefinition:
        return RuleDefinition(
            rule_id="R001",
            name="Required Main Diagnosis",
            description="Main ICD-10 required",
            severity="error",
            condition="icd10_main is not None and len(str(icd10_main)) >= 3",
            on_violation={
                "message": "Missing main diagnosis",
                "autofix_hint": "add_if_absent",
            },
            enabled=True,
            version="1.0",
        )

    def test_execute_rule_sat(self, executor: RuleExecutor, sample_rule: RuleDefinition, sample_claim_dict: dict):
        """Test rule returns SAT for valid claim."""
        result = executor.execute(sample_rule, sample_claim_dict)

        assert result.state == RuleState.SAT

    def test_execute_rule_viol(self, executor: RuleExecutor, sample_rule: RuleDefinition, sample_claim_dict: dict):
        """Test rule returns VIOL for invalid claim."""
        sample_claim_dict["icd10_main"] = None

        result = executor.execute(sample_rule, sample_claim_dict)

        assert result.state == RuleState.VIOL
        assert result.message is not None

    def test_execute_with_claim_record(self, executor: RuleExecutor, sample_rule: RuleDefinition, sample_claim: ClaimRecord):
        """Test execute works with ClaimRecord object."""
        result = executor.execute(sample_rule, sample_claim)

        assert result.state == RuleState.SAT


class TestRuleEngine:
    """Tests for RuleEngine orchestrator."""

    @pytest.fixture
    def engine(self) -> RuleEngine:
        rules_path = Path("config/rules")
        if not rules_path.exists():
            pytest.skip("config/rules not found")
        return RuleEngine(rules_path)

    def test_load_rules(self, engine: RuleEngine):
        """Test engine loads rules."""
        assert len(engine.rules) >= 1

    def test_validate_batch(self, engine: RuleEngine, sample_claim_batch):
        """Test validating a batch returns ValidationReport."""
        report = engine.validate(sample_claim_batch)

        assert hasattr(report, "total_records")
        assert hasattr(report, "violations")
        assert hasattr(report, "pass_rate")

    def test_get_rule(self, engine: RuleEngine):
        """Test getting rule by ID."""
        rule = engine.get_rule("R001")

        if rule:
            assert rule.rule_id == "R001"

    def test_enabled_rules_filter(self, engine: RuleEngine):
        """Test enabled_rules property filters disabled rules."""
        enabled = engine.enabled_rules

        assert all(r.enabled for r in enabled)


class TestRuleDefinition:
    """Tests for RuleDefinition model."""

    def test_create_valid_rule(self):
        """Test creating valid rule definition."""
        rule = RuleDefinition(
            rule_id="R999",
            name="Test Rule",
            description="Test description",
            severity="error",
            condition="True",
            on_violation={"message": "Test error"},
            enabled=True,
            version="1.0",
        )

        assert rule.rule_id == "R999"
        assert rule.severity == "error"

    def test_severity_values(self):
        """Test severity accepts valid values."""
        for severity in ["error", "warning"]:
            rule = RuleDefinition(
                rule_id="R999",
                name="Test",
                description="Test",
                severity=severity,
                condition="True",
                on_violation={"message": "Test"},
            )
            assert rule.severity == severity


class TestRuleResult:
    """Tests for RuleResult model."""

    def test_create_sat_result(self):
        """Test creating SAT result."""
        result = RuleResult(
            rule_id="R001",
            case_id="ENC123",
            state=RuleState.SAT,
        )

        assert result.state == RuleState.SAT

    def test_create_viol_result(self):
        """Test creating VIOL result."""
        result = RuleResult(
            rule_id="R001",
            case_id="ENC123",
            state=RuleState.VIOL,
            message="Missing diagnosis",
        )

        assert result.state == RuleState.VIOL
        assert result.message == "Missing diagnosis"
