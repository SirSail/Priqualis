"""PII Masking for Priqualis."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Any

import polars as pl

logger = logging.getLogger(__name__)

def mask_pesel(value: str) -> str:
    """Mask PESEL with deterministic SHA-256 hash (first 8 chars)."""
    if not value or len(value) != 11:
        return value
    return f"PESEL_{hashlib.sha256(value.encode()).hexdigest()[:8]}"

def mask_name(value: str) -> str:
    if not value: return value
    return f"PAT_{hashlib.sha256(value.encode()).hexdigest()[:8].upper()}"

def mask_address(value: str) -> str:
    return "[MASKED_ADDRESS]" if value else value

def mask_phone(value: str) -> str:
    return "[MASKED_PHONE]" if value else value

def mask_email(value: str) -> str:
    return "[MASKED_EMAIL]" if value else value

# Regex Patterns
PESEL_PATTERN = re.compile(r"\b\d{11}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+48\s?)?(?:\d{3}[\s-]?\d{3}[\s-]?\d{3}|\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Protected fields
DEFAULT_PROTECTED_FIELDS: frozenset[str] = frozenset({
    "case_id", "patient_id", "pesel_masked", "jgp_code",
    "department_code", "icd10_main", "icd10_secondary", "procedures",
})

@dataclass(slots=True, frozen=True)
class MaskingRule:
    field_name: str
    mask_fn: Callable[[str], str]
    description: str = ""

DEFAULT_MASKING_RULES = (
    MaskingRule("pesel", mask_pesel),
    MaskingRule("pesel_raw", mask_pesel),
    MaskingRule("patient_name", mask_name),
    MaskingRule("name", mask_name),
    MaskingRule("surname", mask_name),
    MaskingRule("address", mask_address),
    MaskingRule("phone", mask_phone),
    MaskingRule("email", mask_email),
)


@dataclass(slots=True)
class PIIMasker:
    """Masks PII fields in claim data."""
    
    field_rules: tuple[MaskingRule, ...] = DEFAULT_MASKING_RULES
    scan_content: bool = True
    protected_fields: frozenset[str] = DEFAULT_PROTECTED_FIELDS

    def mask_dataframe(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """
        Mask PII in DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (masked DataFrame, count of masked fields)
        """
        masked_count = 0
        result = df.clone()

        # Build lowercase -> actual column name map
        col_name_map = {col.lower(): col for col in result.columns}

        # Apply field-level masking
        for rule in self.field_rules:
            actual_col = col_name_map.get(rule.field_name.lower())

            # Guard clauses
            if actual_col is None:
                continue
            if actual_col in self.protected_fields:
                continue

            result = result.with_columns(
                pl.col(actual_col)
                .map_elements(rule.mask_fn, return_dtype=pl.Utf8)
                .alias(actual_col)
            )
            masked_count += 1
            logger.debug("Masked field '%s' with %s", actual_col, rule.description)

        # Scan text content for PII patterns
        if self.scan_content:
            result = self._mask_text_columns(result)

        return result, masked_count

    def _mask_text_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply regex-based masking to text columns."""
        result = df

        for col in df.columns:
            # Guard clauses
            if col in self.protected_fields:
                continue
            if df[col].dtype != pl.Utf8:
                continue

            result = result.with_columns(
                pl.col(col)
                .map_elements(self._mask_content, return_dtype=pl.Utf8)
                .alias(col)
            )

        return result

    def _mask_content(self, text: str | None) -> str | None:
        """Mask PII patterns in text content using regex."""
        if not text:
            return text

        # Apply regex substitutions
        text = PESEL_PATTERN.sub("[MASKED_PESEL]", text)
        text = PHONE_PATTERN.sub("[MASKED_PHONE]", text)
        text = EMAIL_PATTERN.sub("[MASKED_EMAIL]", text)

        return text

    def mask_dict(self, record: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """
        Mask PII in a dictionary record.

        Args:
            record: Input record

        Returns:
            Tuple of (masked record, count of masked fields)
        """
        masked = record.copy()
        masked_count = 0

        for rule in self.field_rules:
            # Guard clauses
            if rule.field_name not in masked:
                continue
            if rule.field_name in self.protected_fields:
                continue

            value = masked[rule.field_name]
            if not value:
                continue

            masked[rule.field_name] = rule.mask_fn(str(value))
            masked_count += 1

        return masked, masked_count
