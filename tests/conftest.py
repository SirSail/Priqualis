"""
Pytest configuration and shared fixtures for Priqualis tests.
"""

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from priqualis.etl.schemas import ClaimRecord, ClaimBatch, AdmissionMode


# =============================================================================
# Paths
# =============================================================================


@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent.parent / "data" / "fixtures"


@pytest.fixture
def tmp_data_path(tmp_path: Path) -> Path:
    """Temporary data directory for test outputs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# =============================================================================
# Sample Data
# =============================================================================


@pytest.fixture
def sample_claim_dict() -> dict[str, Any]:
    """Sample claim as dictionary (matches ClaimRecord schema)."""
    return {
        "case_id": "ENC12345",
        "patient_id": "PAT_MASKED_001",
        "pesel_masked": "PESEL_abc12345",
        "birth_date": "1990-05-15",
        "gender": "M",
        "admission_date": "2024-01-15",
        "discharge_date": "2024-01-18",
        "admission_mode": "emergency",
        "department_code": "4000",
        "jgp_code": "A01",
        "jgp_name": "Test JGP",
        "tariff_value": 3250.0,
        "icd10_main": "J18.9",
        "icd10_secondary": "E11.9,I10",  # String for CSV compatibility
        "procedures": "88.761,99.04",  # String for CSV compatibility
        "status": "new",
        "has_error": False,
        "error_type": None,
    }


@pytest.fixture
def sample_claim(sample_claim_dict: dict) -> ClaimRecord:
    """Sample claim as Pydantic model."""
    return ClaimRecord(**sample_claim_dict)


@pytest.fixture
def sample_batch_dicts(sample_claim_dict: dict) -> list[dict]:
    """Batch of 5 claim dicts with variations."""
    return [
        sample_claim_dict,
        {**sample_claim_dict, "case_id": "ENC12346", "icd10_main": None},  # Missing diagnosis - R001
        {**sample_claim_dict, "case_id": "ENC12347", "jgp_code": "B02"},
        {**sample_claim_dict, "case_id": "ENC12348", "discharge_date": "2024-01-14"},  # Invalid dates - R002
        {**sample_claim_dict, "case_id": "ENC12349", "admission_mode": "planned", "procedures": ""},  # No procedures - R004
    ]


@pytest.fixture
def sample_claim_batch(sample_batch_dicts: list[dict]) -> ClaimBatch:
    """ClaimBatch with sample records."""
    records = []
    for d in sample_batch_dicts:
        try:
            records.append(ClaimRecord(**d))
        except Exception:
            pass  # Skip invalid for this fixture
    return ClaimBatch(records=records)


@pytest.fixture
def sample_claims_df(sample_batch_dicts: list[dict]) -> pl.DataFrame:
    """Sample batch as Polars DataFrame."""
    return pl.DataFrame(sample_batch_dicts)


# =============================================================================
# Rules
# =============================================================================


@pytest.fixture
def sample_rule_yaml() -> str:
    """Sample YAML rule definition matching our format."""
    return """
rules:
  - rule_id: R001
    name: Required Main Diagnosis
    description: Main ICD-10 diagnosis must be present
    severity: error
    condition: icd10_main is not None and len(str(icd10_main)) >= 3
    on_violation:
      message: "Missing or invalid main diagnosis (ICD-10)"
      autofix_hint: add_if_absent
    enabled: true
    version: "1.0"
"""


@pytest.fixture
def sample_rules_dir(tmp_path: Path, sample_rule_yaml: str) -> Path:
    """Temporary directory with sample rules."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "test_rules.yaml").write_text(sample_rule_yaml)
    return rules_dir


# =============================================================================
# Files
# =============================================================================


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_batch_dicts: list[dict]) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_claims.csv"
    df = pl.DataFrame(sample_batch_dicts)
    df.write_csv(csv_path)
    return csv_path


@pytest.fixture
def sample_parquet_file(tmp_path: Path, sample_batch_dicts: list[dict]) -> Path:
    """Create a temporary Parquet file with sample data."""
    parquet_path = tmp_path / "test_claims.parquet"
    df = pl.DataFrame(sample_batch_dicts)
    df.write_parquet(parquet_path)
    return parquet_path