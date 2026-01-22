"""Domain constants for Priqualis."""

from typing import Any

# Maps NFZ error codes to Priqualis rule IDs
NFZ_ERROR_MAPPING: dict[str, str] = {
    "CWV_001": "R001",  # Missing main diagnosis
    "CWV_002": "R002",  # Invalid date range
    "CWV_003": "R003",  # Missing JGP code
    "CWV_010": "R004",  # Procedure mismatch
    "CWV_015": "R005",  # Invalid admission mode
    "CWV_020": "R006",  # Missing department code
    "CWV_025": "R007",  # Invalid tariff
}

# Defaults
DEFAULT_ICD10_PLACEHOLDER: str = "Z00.0"
DEFAULT_JGP_CODE: str = "A01"
DEFAULT_DEPARTMENT_CODE: str = "4000"
DATE_PARSE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")

AUTOFIX_DEFAULT_VALUES: dict[str, Any] = {
    "icd10_main": DEFAULT_ICD10_PLACEHOLDER,
    "icd10_secondary": [],
    "jgp_code": DEFAULT_JGP_CODE,
    "jgp_name": "General Medicine",
    "tariff_value": 0.0,
    "admission_mode": "planned",
    "department_code": DEFAULT_DEPARTMENT_CODE,
    "procedures": [],
}

AUTOFIX_SUGGESTED_FIXES: dict[str, dict[str, Any]] = {
    "R001": {"field": "icd10_main", "value": DEFAULT_ICD10_PLACEHOLDER, "rationale": "Added placeholder diagnosis."},
    "R002": {"field": "discharge_date", "value": None, "rationale": "Discharge date set to admission date."},
    "R003": {"field": "jgp_code", "value": DEFAULT_JGP_CODE, "rationale": "Added default JGP."},
    "R005": {"field": "admission_mode", "value": "planned", "rationale": "Set admission mode to planned."},
    "R006": {"field": "department_code", "value": DEFAULT_DEPARTMENT_CODE, "rationale": "Added default department."},
}
