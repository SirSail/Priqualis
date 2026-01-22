"""ETL Processor for Priqualis."""

import logging, time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import polars as pl
from priqualis.core.config import get_settings
from priqualis.core.exceptions import ETLError
from priqualis.etl.importers import ClaimImporter
from priqualis.etl.pii_masking import PIIMasker
from priqualis.etl.schemas import ClaimBatch, ProcessedBatch

logger = logging.getLogger(__name__)

class OutputFormat(str, Enum):
    PARQUET = "parquet"; CSV = "csv"

@dataclass(slots=True)
class ProcessorConfig:
    mask_pii: bool = True
    scan_content_for_pii: bool = True
    strict_validation: bool = False
    output_format: OutputFormat = OutputFormat.PARQUET
    output_dir: Path | None = None
    batch_size: int = 10000

class ETLProcessor:
    """Orchestrates ETL pipeline: import -> mask -> validate -> export."""

    def __init__(self, config: ProcessorConfig | None = None, *, importer: ClaimImporter | None = None, masker: PIIMasker | None = None):
        self.config = config or ProcessorConfig()
        self.importer = importer or ClaimImporter(strict=self.config.strict_validation)
        self.masker = masker or PIIMasker(scan_content=self.config.scan_content_for_pii)
        self._output_dir = self.config.output_dir or get_settings().data_processed_path

    def process(self, file_path: Path | str) -> ProcessedBatch:
        fp, start = Path(file_path), time.perf_counter()
        logger.info("Starting ETL for %s", fp)

        raw, masked = self.importer.load(fp), 0
        if self.config.mask_pii: raw, masked = self.masker.mask_dataframe(raw)
        
        batch = self.importer.validate(raw, source_file=str(fp))
        logger.info("Validated %d records (%d valid)", batch.count, batch.valid_count)

        out_path = self._save(batch, fp)
        logger.info("ETL completed in %.2f ms", (ms := (time.perf_counter()-start)*1000))
        
        return ProcessedBatch(batch, str(out_path) if out_path else None, ms, masked, self.importer.validation_errors)

    def _save(self, batch: ClaimBatch, input_path: Path) -> Path | None:
        if not self._output_dir: return None
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".parquet" if self.config.output_format == OutputFormat.PARQUET else ".csv"
        out_path = self._output_dir / f"{input_path.stem}_processed_{ts}{ext}"

        df = pl.DataFrame([r.model_dump() for r in batch.records])
        for c in ("icd10_secondary", "procedures"):
            if c in df.columns: df = df.with_columns(pl.col(c).list.join(",").alias(c))

        try:
            if self.config.output_format == OutputFormat.PARQUET: df.write_parquet(out_path)
            else: df.write_csv(out_path)
            return out_path
        except Exception as e: raise ETLError(f"Failed to save batch: {e}") from e

    def validate_only(self, file_path: Path | str) -> ClaimBatch:
        fp = Path(file_path)
        return self.importer.validate(self.importer.load(fp), source_file=str(fp))

def process_file(file_path: Path | str, output_dir: Path | str | None = None, mask_pii: bool = True, output_format: OutputFormat = OutputFormat.PARQUET) -> ProcessedBatch:
    return ETLProcessor(ProcessorConfig(mask_pii=mask_pii, output_dir=Path(output_dir) if output_dir else None, output_format=output_format)).process(file_path)
