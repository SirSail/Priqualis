"""
ETL Processor for Priqualis.

Orchestrates the full ETL pipeline: import → mask → validate → export.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

import polars as pl

from priqualis.core.config import get_settings
from priqualis.core.exceptions import ETLError
from priqualis.etl.importers import ClaimImporter
from priqualis.etl.pii_masking import PIIMasker
from priqualis.etl.schemas import ClaimBatch, ProcessedBatch

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class OutputFormat(str, Enum):
    """Supported output formats."""

    PARQUET = "parquet"
    CSV = "csv"


# =============================================================================
# Protocols (Dependency Inversion)
# =============================================================================


class FileLoader(Protocol):
    """Protocol for file loading."""

    def load(self, file_path: Path | str) -> pl.DataFrame:
        """Load file into DataFrame."""
        ...

    def validate(self, df: pl.DataFrame, source_file: str | None = None) -> ClaimBatch:
        """Validate DataFrame against schema."""
        ...

    @property
    def validation_errors(self) -> list[str]:
        """Get validation errors from last import."""
        ...


class DataMasker(Protocol):
    """Protocol for data masking."""

    def mask_dataframe(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """Mask PII in DataFrame, return (masked_df, count)."""
        ...


# =============================================================================
# Processor Configuration
# =============================================================================


@dataclass(slots=True)
class ProcessorConfig:
    """
    Immutable configuration for ETL processor.

    Note: output_dir defaults to None and is resolved at runtime from settings.
    """

    # Masking
    mask_pii: bool = True
    scan_content_for_pii: bool = True

    # Validation
    strict_validation: bool = False

    # Output
    output_format: OutputFormat = OutputFormat.PARQUET
    output_dir: Path | None = None

    # Performance
    batch_size: int = 10000


# =============================================================================
# Main Processor
# =============================================================================


class ETLProcessor:
    """
    Orchestrates the full ETL pipeline for claim data.

    Pipeline stages:
    1. Import: Load raw data from file
    2. Mask: Apply PII masking
    3. Validate: Validate against schema
    4. Export: Save to Parquet/CSV

    Supports Dependency Injection via constructor for testability.

    Example:
        processor = ETLProcessor()
        result = processor.process("data/raw/claims.csv")
        print(f"Processed {result.batch.count} records")
    """

    def __init__(
        self,
        config: ProcessorConfig | None = None,
        *,
        importer: FileLoader | None = None,
        masker: DataMasker | None = None,
    ):
        """
        Initialize processor with configuration and optional DI.

        Args:
            config: Processor configuration (uses defaults if None)
            importer: Custom file loader (uses ClaimImporter if None)
            masker: Custom data masker (uses PIIMasker if None)
        """
        self.config = config or ProcessorConfig()

        # Dependency Injection - use provided or create defaults
        self.importer: FileLoader = importer or ClaimImporter(
            strict=self.config.strict_validation
        )
        self.masker: DataMasker = masker or PIIMasker(
            scan_content=self.config.scan_content_for_pii
        )

        # Resolve output_dir from settings if not provided
        self._output_dir = self._resolve_output_dir()

    def _resolve_output_dir(self) -> Path | None:
        """Resolve output directory, using settings as fallback."""
        if self.config.output_dir is not None:
            return self.config.output_dir

        settings = get_settings()
        return settings.data_processed_path

    def process(self, file_path: Path | str) -> ProcessedBatch:
        """
        Run full ETL pipeline on input file.

        Args:
            file_path: Path to input file

        Returns:
            ProcessedBatch with validated records and metadata

        Raises:
            ETLError: If processing fails
        """
        file_path = Path(file_path)
        start_time = time.perf_counter()

        logger.info("Starting ETL pipeline for %s", file_path)

        # Stage 1: Import
        raw_df = self.importer.load(file_path)
        logger.debug("Loaded %d rows from %s", len(raw_df), file_path)

        # Stage 2: Mask PII
        pii_masked_count = 0
        if self.config.mask_pii:
            raw_df, pii_masked_count = self.masker.mask_dataframe(raw_df)
            logger.debug("Masked %d PII fields", pii_masked_count)

        # Stage 3: Validate
        batch = self.importer.validate(raw_df, source_file=str(file_path))
        logger.info(
            "Validated %d records (%d valid, %d with errors)",
            batch.count,
            batch.valid_count,
            batch.error_count,
        )

        # Stage 4: Generate output path
        output_path = self._generate_output_path(file_path)

        # Stage 5: Export
        if output_path:
            self._save_batch(batch, output_path)
            logger.info("Saved processed batch to %s", output_path)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info("ETL pipeline completed in %.2f ms", processing_time_ms)

        return ProcessedBatch(
            batch=batch,
            output_path=str(output_path) if output_path else None,
            processing_time_ms=processing_time_ms,
            pii_fields_masked=pii_masked_count,
            schema_errors=self.importer.validation_errors,
        )

    def _generate_output_path(self, input_path: Path) -> Path | None:
        """Generate output file path with timestamp."""
        # Guard clause
        if not self._output_dir:
            return None

        self._output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = input_path.stem
        suffix = ".parquet" if self.config.output_format == OutputFormat.PARQUET else ".csv"

        return self._output_dir / f"{stem}_processed_{timestamp}{suffix}"

    def _save_batch(self, batch: ClaimBatch, output_path: Path) -> None:
        """Save batch to file."""
        # Convert to DataFrame
        records_dict = [r.model_dump() for r in batch.records]
        df = pl.DataFrame(records_dict)

        # Flatten list columns for serialization
        for col in ("icd10_secondary", "procedures"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).list.join(",").alias(col))

        try:
            if self.config.output_format == OutputFormat.PARQUET:
                df.write_parquet(output_path)
            else:
                df.write_csv(output_path)
        except Exception as e:
            raise ETLError(
                f"Failed to save batch to {output_path}: {e}. "
                f"Check write permissions and disk space."
            ) from e

    def validate_only(self, file_path: Path | str) -> ClaimBatch:
        """
        Load and validate without masking or saving.

        Useful for quick validation checks.

        Args:
            file_path: Path to input file

        Returns:
            ClaimBatch with validated records
        """
        file_path = Path(file_path)
        df = self.importer.load(file_path)
        return self.importer.validate(df, source_file=str(file_path))


# =============================================================================
# Convenience Functions
# =============================================================================


def process_file(
    file_path: Path | str,
    output_dir: Path | str | None = None,
    mask_pii: bool = True,
    output_format: OutputFormat = OutputFormat.PARQUET,
) -> ProcessedBatch:
    """
    Convenience function for processing a single file.

    Args:
        file_path: Path to input file
        output_dir: Output directory (optional)
        mask_pii: Whether to mask PII
        output_format: Output format (parquet or csv)

    Returns:
        ProcessedBatch with results
    """
    config = ProcessorConfig(
        mask_pii=mask_pii,
        output_dir=Path(output_dir) if output_dir else None,
        output_format=output_format,
    )
    processor = ETLProcessor(config)
    return processor.process(file_path)
