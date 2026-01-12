"""
ETL Importers for Priqualis.

Loads claim data from various file formats (CSV, XML, Parquet).
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Protocol

import polars as pl
from pydantic import ValidationError

from priqualis.core.exceptions import ETLError, SchemaValidationError
from priqualis.etl.schemas import ClaimBatch, ClaimRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class FileFormat(str, Enum):
    """Supported file formats for import."""

    CSV = "csv"
    PARQUET = "parquet"
    XML = "xml"


# =============================================================================
# Protocols
# =============================================================================


class RecordValidator(Protocol):
    """Protocol for record validation."""

    def __call__(self, **kwargs) -> ClaimRecord:
        """Validate and create a claim record."""
        ...


# =============================================================================
# File Format Detection
# =============================================================================


# Extension to format mapping (immutable)
EXTENSION_FORMAT_MAP: dict[str, FileFormat] = {
    ".csv": FileFormat.CSV,
    ".parquet": FileFormat.PARQUET,
    ".pq": FileFormat.PARQUET,
    ".xml": FileFormat.XML,
}


def detect_format(file_path: Path) -> FileFormat:
    """
    Detect file format from extension.

    Args:
        file_path: Path to the input file

    Returns:
        Detected FileFormat enum

    Raises:
        ETLError: If format is not supported
    """
    suffix = file_path.suffix.lower()
    file_format = EXTENSION_FORMAT_MAP.get(suffix)

    if file_format is None:
        supported = ", ".join(EXTENSION_FORMAT_MAP.keys())
        raise ETLError(
            f"Unsupported file format '{suffix}' for {file_path}. "
            f"Supported extensions: {supported}"
        )

    return file_format


# =============================================================================
# Loaders
# =============================================================================


def load_csv(file_path: Path, separator: str | None = None) -> pl.DataFrame:
    """
    Load CSV file with auto-detected separator.

    Args:
        file_path: Path to CSV file
        separator: Optional separator (auto-detect if None)

    Returns:
        Polars DataFrame

    Raises:
        ETLError: If loading fails
    """
    try:
        # Fast path: explicit separator
        if separator:
            return pl.read_csv(file_path, separator=separator)

        # Auto-detect separator by trying common ones
        for sep in (",", ";", "\t", "|"):
            try:
                df = pl.read_csv(file_path, separator=sep)
                if len(df.columns) > 1:
                    logger.debug("Auto-detected separator '%s' for %s", sep, file_path)
                    return df
            except Exception:
                continue

        # Fallback to comma
        return pl.read_csv(file_path)

    except Exception as e:
        raise ETLError(
            f"Failed to load CSV file {file_path}: {e}. "
            "Check file encoding and format."
        ) from e


def load_parquet(file_path: Path) -> pl.DataFrame:
    """
    Load Parquet file.

    Args:
        file_path: Path to Parquet file

    Returns:
        Polars DataFrame

    Raises:
        ETLError: If loading fails
    """
    try:
        return pl.read_parquet(file_path)
    except Exception as e:
        raise ETLError(
            f"Failed to load Parquet file {file_path}: {e}. "
            "Check file integrity and Parquet version."
        ) from e


def load_xml(file_path: Path) -> pl.DataFrame:
    """
    Load XML file (NFZ SWIAD format stub).

    Args:
        file_path: Path to XML file

    Returns:
        Polars DataFrame

    Raises:
        NotImplementedError: XML parsing not yet implemented

    Note:
        This is a placeholder. Full XML parsing for NFZ SWIAD format
        requires specific schema handling.
    """
    raise NotImplementedError(
        f"XML import not yet implemented for {file_path}. "
        "Use CSV or Parquet format. "
        "XML support planned for v0.2.0."
    )


# Format to loader function mapping
FORMAT_LOADERS: dict[FileFormat, callable] = {
    FileFormat.CSV: load_csv,
    FileFormat.PARQUET: load_parquet,
    FileFormat.XML: load_xml,
}


# =============================================================================
# Main Importer
# =============================================================================


class ClaimImporter:
    """
    Imports claim data from files and validates against schema.

    Supports CSV (with auto-detected separator), Parquet, and XML (stub).

    Implements FileLoader protocol for DI compatibility.
    """

    def __init__(self, strict: bool = False):
        """
        Initialize importer.

        Args:
            strict: If True, raise on first validation error.
                   If False, collect all errors and skip invalid records.
        """
        self.strict = strict
        self._validation_errors: list[str] = []

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors from last import (immutable copy)."""
        return self._validation_errors.copy()

    def load(self, file_path: Path | str) -> pl.DataFrame:
        """
        Load raw data from file.

        Args:
            file_path: Path to input file

        Returns:
            Polars DataFrame with raw data

        Raises:
            ETLError: If file not found or loading fails
        """
        file_path = Path(file_path)

        # Guard clause: file must exist
        if not file_path.exists():
            raise ETLError(
                f"File not found: {file_path}. "
                "Check path and permissions."
            )

        file_format = detect_format(file_path)
        loader = FORMAT_LOADERS[file_format]

        logger.debug("Loading %s file: %s", file_format.value, file_path)
        return loader(file_path)

    def validate(
        self,
        df: pl.DataFrame,
        source_file: str | None = None,
    ) -> ClaimBatch:
        """
        Validate DataFrame against ClaimRecord schema.

        Args:
            df: Raw DataFrame
            source_file: Optional source file path for metadata

        Returns:
            ClaimBatch with validated records

        Raises:
            SchemaValidationError: If strict mode and validation fails,
                                   or if all records fail validation
        """
        self._validation_errors = []
        records: list[ClaimRecord] = []
        total_rows = len(df)

        for row_idx, row in enumerate(df.iter_rows(named=True)):
            try:
                record = ClaimRecord(**row)
                records.append(record)
            except ValidationError as e:
                error_msg = f"Row {row_idx}: {e.error_count()} validation errors"
                self._validation_errors.append(error_msg)

                # Log detailed error
                logger.warning(
                    "Validation failed for row %d: %s",
                    row_idx,
                    e.errors()[:3],  # First 3 errors
                )

                # Early exit in strict mode
                if self.strict:
                    raise SchemaValidationError(
                        f"Validation failed at row {row_idx}: {e}. "
                        "Set strict=False to skip invalid records."
                    ) from e

        # Guard: at least some records must be valid
        if self._validation_errors and not records:
            raise SchemaValidationError(
                f"All {len(self._validation_errors)} of {total_rows} records "
                f"failed validation. Check data format and schema."
            )

        if self._validation_errors:
            logger.warning(
                "%d of %d records failed validation. Skipped.",
                len(self._validation_errors),
                total_rows,
            )

        return ClaimBatch(
            records=records,
            source_file=source_file,
        )

    def import_file(self, file_path: Path | str) -> ClaimBatch:
        """
        Load and validate file in one step.

        Args:
            file_path: Path to input file

        Returns:
            ClaimBatch with validated records
        """
        file_path = Path(file_path)
        df = self.load(file_path)
        return self.validate(df, source_file=str(file_path))
