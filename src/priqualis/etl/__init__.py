"""
ETL module for Priqualis.

Provides data import, validation, PII masking, and processing capabilities.
"""

from priqualis.etl.importers import ClaimImporter, FileFormat
from priqualis.etl.pii_masking import (
    DEFAULT_PROTECTED_FIELDS,
    MaskingRule,
    PIIMasker,
    mask_name,
    mask_pesel,
)
from priqualis.etl.processor import (
    ETLProcessor,
    OutputFormat,
    ProcessorConfig,
    process_file,
)
from priqualis.etl.schemas import (
    AdmissionMode,
    ClaimBatch,
    ClaimRecord,
    ClaimStatus,
    DiagnosisInfo,
    Gender,
    JGPInfo,
    PatientInfo,
    ProcedureInfo,
    ProcessedBatch,
)

__all__ = [
    # Schemas
    "AdmissionMode",
    "ClaimBatch",
    "ClaimRecord",
    "ClaimStatus",
    "DiagnosisInfo",
    "Gender",
    "JGPInfo",
    "PatientInfo",
    "ProcedureInfo",
    "ProcessedBatch",
    # Enums
    "FileFormat",
    "OutputFormat",
    # Importer
    "ClaimImporter",
    # Masking
    "PIIMasker",
    "MaskingRule",
    "DEFAULT_PROTECTED_FIELDS",
    "mask_pesel",
    "mask_name",
    # Processor
    "ETLProcessor",
    "ProcessorConfig",
    "process_file",
]