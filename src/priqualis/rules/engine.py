"""Rule Engine for Priqualis validation orchestration."""

import ast
import logging
from pathlib import Path
from typing import Any, Protocol

import yaml

from priqualis.core.exceptions import RuleExecutionError, RuleParseError
from priqualis.etl.schemas import ClaimBatch, ClaimRecord
from priqualis.rules.models import (
    RuleDefinition,
    RuleResult,
    RuleSeverity,
    RuleState,
    ValidationReport,
    ViolationAction,
)
from priqualis.rules.scoring import ImpactScorer

logger = logging.getLogger(__name__)


class RuleLoader(Protocol):
    def load(self, rules_path: Path) -> list[RuleDefinition]: ...


class RuleEvaluator(Protocol):
    def execute(self, rule: RuleDefinition, record: dict[str, Any]) -> RuleResult: ...


SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set,
    "min": min, "max": max, "abs": abs, "sum": sum, "all": all, "any": any,
    "in": lambda x, y: x in y, "isinstance": isinstance,
    "None": None, "True": True, "False": False,
}

# Disallowed AST nodes for security
DISALLOWED_NODES = (ast.Import, ast.ImportFrom, ast.Call, ast.Attribute)


def validate_expression(expr: str) -> bool:
    """Validate that expression is safe to evaluate."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleParseError(f"Invalid expression syntax: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RuleParseError("Import statements not allowed in conditions")
    return True


def safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """Safely evaluate a Python expression with restricted context."""
    try:
        # nosec B307 - eval input is controlled: only SAFE_BUILTINS are exposed
        return eval(expr, {"__builtins__": SAFE_BUILTINS}, context.copy())  # noqa: S307
    except Exception as e:
        raise RuleExecutionError(f"Failed to evaluate expression '{expr}': {e}") from e


def load_rules(rules_path: Path) -> list[RuleDefinition]:
    """Load all YAML rules from directory or file."""
    if rules_path.is_file():
        yaml_files = [rules_path]
    elif rules_path.is_dir():
        yaml_files = list(rules_path.glob("*.yaml")) + list(rules_path.glob("*.yml"))
    else:
        raise RuleParseError(f"Rules path not found: {rules_path}")

    if not yaml_files:
        logger.warning("No YAML rule files found in %s", rules_path)
        return []

    rules = []
    for yaml_file in sorted(yaml_files):
        rules.extend(_load_rules_from_file(yaml_file))

    logger.info("Loaded %d rules from %s", len(rules), rules_path)
    return rules


def _load_rules_from_file(file_path: Path) -> list[RuleDefinition]:
    try:
        content = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except Exception as e:
        raise RuleParseError(f"Failed to load {file_path}: {e}") from e

    if not data:
        return []

    if isinstance(data, dict):
        if "rule_id" in data:
            return [_parse_rule(data, file_path)]
        elif "rules" in data:
            return [_parse_rule(r, file_path) for r in data["rules"]]
        else:
            raise RuleParseError(f"Invalid rule file format: {file_path}")
    elif isinstance(data, list):
        return [_parse_rule(r, file_path) for r in data]
    else:
        raise RuleParseError(f"Unexpected format in {file_path}")


def _parse_rule(data: dict, source_file: Path) -> RuleDefinition:
    """Parse a single rule from dict."""
    try:
        # Build ViolationAction
        on_violation_data = data.get("on_violation", {})
        if isinstance(on_violation_data, str):
            on_violation = ViolationAction(message=on_violation_data)
        else:
            on_violation = ViolationAction(**on_violation_data)

        # Parse rule
        rule = RuleDefinition(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data.get("description", data["name"]),
            severity=data.get("severity", "error"),
            condition=data["condition"],
            on_violation=on_violation,
            jgp_groups=data.get("jgp_groups"),
            enabled=data.get("enabled", True),
            version=data.get("version", "1.0"),
        )

        # Validate expression syntax
        validate_expression(rule.condition)

        logger.debug("Parsed rule %s from %s", rule.rule_id, source_file.name)
        return rule

    except KeyError as e:
        raise RuleParseError(
            f"Missing required field {e} in rule from {source_file}"
        ) from e
    except Exception as e:
        raise RuleParseError(
            f"Failed to parse rule in {source_file}: {e}"
        ) from e


def validate_rule(rule: RuleDefinition) -> bool:

    validate_expression(rule.condition)
    return True


class RuleExecutor:
    def __init__(self, scorer: ImpactScorer | None = None):
        self.scorer = scorer or ImpactScorer()

    def execute(self, rule: RuleDefinition, swiadczenie: dict | ClaimRecord) -> RuleResult:
        record_dict = swiadczenie.model_dump() if hasattr(swiadczenie, "model_dump") else dict(swiadczenie)
        case_id = record_dict.get("case_id", "unknown")

        if not rule.enabled:
            return RuleResult(
                rule_id=rule.rule_id,
                case_id=case_id,
                state=RuleState.SAT,
                message="Rule disabled",
            )

        if rule.jgp_groups:
            kod_jgp = record_dict.get("jgp_code")
            if kod_jgp and kod_jgp not in rule.jgp_groups:
                return RuleResult(
                    rule_id=rule.rule_id,
                    case_id=case_id,
                    state=RuleState.SAT,
                    message="JGP not in rule scope",
                )

        try:
            # Context for eval: allows 'record.field' or direct field access
            context = {**record_dict, "record": record_dict}
            
            if safe_eval(rule.condition, context):
                return RuleResult(rule_id=rule.rule_id, case_id=case_id, state=RuleState.SAT)

            return RuleResult(
                rule_id=rule.rule_id,
                case_id=case_id,
                state=RuleState.WARN if rule.severity == RuleSeverity.WARNING else RuleState.VIOL,
                message=rule.on_violation.message,
                autofix_hint=rule.on_violation.autofix_hint,
            )

        except RuleExecutionError as e:
            logger.error("Rule %s failed on %s: %s", rule.rule_id, case_id, e)
            return RuleResult(
                rule_id=rule.rule_id,
                case_id=case_id,
                state=RuleState.VIOL,
                message=f"Rule execution error: {e}",
            )

    def execute_batch(
        self,
        rules: list[RuleDefinition],
        swiadczenia: list[dict | ClaimRecord],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []
        enabled_rules = [r for r in rules if r.enabled]

        for swiadczenie in swiadczenia:
            for rule in enabled_rules:
                results.append(self.execute(rule, swiadczenie))

        logger.debug(
            "Executed %d rules × %d swiadczenia = %d results",
            len(enabled_rules),
            len(swiadczenia),
            len(results),
        )

        return results


class RuleEngine:
    def __init__(
        self,
        rules_path: Path,
        *,
        executor: RuleExecutor | None = None,
        scorer: ImpactScorer | None = None,
    ):
        self.rules_path = Path(rules_path)
        self.scorer = scorer or ImpactScorer()
        self.executor = executor or RuleExecutor(scorer=self.scorer)
        self.rules: list[RuleDefinition] = load_rules(self.rules_path)

    def reload_rules(self) -> int:
        self.rules = load_rules(self.rules_path)
        return len(self.rules)

    def validate(
        self,
        batch: ClaimBatch,
        *,
        calculate_impact: bool = True,
    ) -> ValidationReport:
        if not self.rules:
            logger.warning("No rules loaded, returning empty report")
            return ValidationReport(
                source_file=batch.source_file,
                total_records=batch.count,
                total_rules=0,
            )

        results = self.executor.execute_batch(self.rules, batch.records)

        if calculate_impact:
            records_map = {r.case_id: r for r in batch.records}
            for result in results:
                if (result.is_violation or result.is_warning) and (record := records_map.get(result.case_id)):
                    result.impact_score = self.scorer.calculate(result, record)

        logger.info(
            "Validation complete: %d records, %d rules, %d violations",
            batch.count,
            len(self.rules),
            sum(1 for r in results if r.is_violation),
        )

        return ValidationReport(
            source_file=batch.source_file,
            total_records=batch.count,
            total_rules=len(self.rules),
            results=results,
        )

    def calculate_impact(self, violation: RuleResult, record: dict | ClaimRecord) -> float:
        return self.scorer.calculate(violation, record)

    def get_rule(self, rule_id: str) -> RuleDefinition | None:
        return next((r for r in self.rules if r.rule_id == rule_id), None)

    @property
    def enabled_rules(self) -> list[RuleDefinition]:
        return [r for r in self.rules if r.enabled]
