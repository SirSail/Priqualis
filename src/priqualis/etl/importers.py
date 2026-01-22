"""ETL Importers for Priqualis."""

import logging
from enum import Enum
from pathlib import Path
import polars as pl
from pydantic import ValidationError
from priqualis.core.exceptions import ETLError, SchemaValidationError
from priqualis.etl.schemas import ClaimBatch, ClaimRecord

logger = logging.getLogger(__name__)

class FileFormat(str, Enum):
    CSV = "csv"; PARQUET = "parquet"; XML = "xml"

EXTENSION_FORMAT_MAP = {".csv": FileFormat.CSV, ".parquet": FileFormat.PARQUET, ".pq": FileFormat.PARQUET, ".xml": FileFormat.XML}

def detect_format(file_path: Path) -> FileFormat:
    if fmt := EXTENSION_FORMAT_MAP.get(file_path.suffix.lower()): return fmt
    raise ETLError(f"Unsupported format '{file_path.suffix}' for {file_path}")

def load_csv(file_path: Path, separator: str | None = None) -> pl.DataFrame:
    try:
        if separator: return pl.read_csv(file_path, separator=separator)
        for sep in (",", ";", "\t", "|"):
            try:
                if len((df := pl.read_csv(file_path, separator=sep)).columns) > 1: return df
            except: continue
        return pl.read_csv(file_path)
    except Exception as e: raise ETLError(f"Failed to load CSV {file_path}: {e}") from e

def load_parquet(file_path: Path) -> pl.DataFrame:
    try: return pl.read_parquet(file_path)
    except Exception as e: raise ETLError(f"Failed to load Parquet {file_path}: {e}") from e

def load_xml(file_path: Path) -> pl.DataFrame:
    raise NotImplementedError(f"XML import not implemented for {file_path}")

FORMAT_LOADERS = {FileFormat.CSV: load_csv, FileFormat.PARQUET: load_parquet, FileFormat.XML: load_xml}

class ClaimImporter:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self._validation_errors = []

    @property
    def validation_errors(self) -> list[str]: return self._validation_errors.copy()

    def load(self, file_path: Path | str) -> pl.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists(): raise ETLError(f"File not found: {file_path}")
        return FORMAT_LOADERS[detect_format(file_path)](file_path)

    def validate(self, df: pl.DataFrame, source_file: str | None = None) -> ClaimBatch:
        self._validation_errors, records = [], []
        for i, row in enumerate(df.iter_rows(named=True)):
            try: records.append(ClaimRecord(**row))
            except ValidationError as e:
                self._validation_errors.append(f"Row {i}: {e.error_count()} errors")
                if self.strict: raise SchemaValidationError(f"Validation failed at row {i}: {e}") from e
        
        if self._validation_errors and not records: raise SchemaValidationError(f"All {len(self.df)} records failed validation")
        if self._validation_errors: logger.warning("%d records failed validation", len(self._validation_errors))
        return ClaimBatch(records=records, source_file=source_file)

    def import_file(self, file_path: Path | str) -> ClaimBatch:
        path = Path(file_path)
        return self.validate(self.load(path), source_file=str(path))
