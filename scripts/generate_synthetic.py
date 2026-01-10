#!/usr/bin/env python3
"""
Synthetic Data Generator for Priqualis.

Generates realistic NFZ-style claim records with controlled error injection
for testing the validation pipeline.

Usage:
    python scripts/generate_synthetic.py --output data/raw/claims.parquet --count 10000 --error-rate 0.2
"""

import argparse
import hashlib
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

# =============================================================================
# Reference Data (Polish healthcare codes)
# =============================================================================

# ICD-10 codes commonly used in Polish hospitals
ICD10_CODES = [
    # Respiratory
    "J18.9",  # Pneumonia
    "J44.1",  # COPD with acute exacerbation
    "J45.0",  # Asthma
    "J06.9",  # Upper respiratory infection
    # Cardiovascular
    "I10",    # Hypertension
    "I21.0",  # Acute MI - anterior wall
    "I21.1",  # Acute MI - inferior wall
    "I25.1",  # Atherosclerotic heart disease
    "I50.0",  # Heart failure
    "I63.9",  # Cerebral infarction
    # Endocrine
    "E11.9",  # Type 2 diabetes
    "E10.9",  # Type 1 diabetes
    "E03.9",  # Hypothyroidism
    # Digestive
    "K80.2",  # Gallstones with cholecystitis
    "K35.8",  # Acute appendicitis
    "K25.0",  # Gastric ulcer with hemorrhage
    "K29.7",  # Gastritis
    # Musculoskeletal
    "M54.5",  # Low back pain
    "M17.1",  # Primary gonarthrosis
    "S72.0",  # Femoral neck fracture
    "S82.0",  # Tibial fracture
    # Neoplasms
    "C34.1",  # Lung cancer
    "C50.9",  # Breast cancer
    "C18.9",  # Colon cancer
    # Genitourinary
    "N39.0",  # Urinary tract infection
    "N18.9",  # Chronic kidney disease
]

# JGP codes (Jednorodne Grupy Pacjentów - Polish DRG)
# Names in English for international collaboration (Polish originals in comments)
JGP_CODES = {
    # Pulmonology
    "A01": {"name": "Pneumonia", "tariff": 3250.0, "procedures": ["88.761", "99.04"]},  # Zapalenie płuc
    "A02": {"name": "COPD Exacerbation", "tariff": 2800.0, "procedures": ["93.94", "99.04"]},  # POChP z zaostrzeniem
    "A03": {"name": "Bronchial Asthma", "tariff": 1950.0, "procedures": ["93.94"]},  # Astma oskrzelowa
    # Cardiology
    "B01": {"name": "Myocardial Infarction", "tariff": 8500.0, "procedures": ["37.22", "88.53", "99.04"]},  # Zawał serca
    "B02": {"name": "Heart Failure", "tariff": 4200.0, "procedures": ["88.72", "89.52"]},  # Niewydolność serca
    "B03": {"name": "Catheter Ablation", "tariff": 12500.0, "procedures": ["37.34", "88.53"]},  # Ablacja
    # Surgery
    "C01": {"name": "Cholecystectomy", "tariff": 5800.0, "procedures": ["51.23", "88.76"]},  # Cholecystektomia
    "C02": {"name": "Appendectomy", "tariff": 4500.0, "procedures": ["47.09", "88.76"]},  # Appendektomia
    "C03": {"name": "Hernia Repair", "tariff": 3800.0, "procedures": ["53.00"]},  # Hernioplastyka
    # Orthopedics
    "D01": {"name": "Total Hip Replacement", "tariff": 18500.0, "procedures": ["81.51", "88.26"]},  # Endoproteza biodra
    "D02": {"name": "Total Knee Replacement", "tariff": 17200.0, "procedures": ["81.54", "88.26"]},  # Endoproteza kolana
    "D03": {"name": "Femoral Fracture", "tariff": 9800.0, "procedures": ["79.35", "88.26"]},  # Złamanie kości udowej
    # Internal medicine
    "E01": {"name": "Diabetes Hospitalization", "tariff": 2100.0, "procedures": ["99.04"]},  # Cukrzyca - hospitalizacja
    "E02": {"name": "Hypertension Workup", "tariff": 1800.0, "procedures": ["89.52", "88.72"]},  # Nadciśnienie - diagnostyka
    # Oncology
    "F01": {"name": "Chemotherapy", "tariff": 6500.0, "procedures": ["99.25"]},  # Chemioterapia
    "F02": {"name": "Radiotherapy", "tariff": 8200.0, "procedures": ["92.29"]},  # Radioterapia
}

# Procedure codes (ICD-9-CM used in Poland)
PROCEDURE_CODES = [
    "37.22",  # Left heart catheterization
    "37.34",  # Catheter ablation
    "47.09",  # Appendectomy
    "51.23",  # Laparoscopic cholecystectomy
    "53.00",  # Hernia repair
    "79.35",  # Open reduction of fracture
    "81.51",  # Total hip replacement
    "81.54",  # Total knee replacement
    "88.26",  # Skeletal X-ray
    "88.53",  # Coronary angiography
    "88.72",  # Echocardiography
    "88.76",  # Abdominal ultrasound
    "88.761", # Abdominal CT
    "89.52",  # ECG
    "92.29",  # Radiotherapy
    "93.94",  # Respiratory therapy
    "99.04",  # Transfusion
    "99.25",  # Chemotherapy injection
]

# Department codes (NFZ internal)
DEPARTMENT_CODES = [
    "4000",  # Internal medicine
    "4100",  # Cardiology
    "4200",  # Gastroenterology
    "4500",  # Pulmonology
    "4600",  # Nephrology
    "5000",  # General surgery
    "5100",  # Orthopedics
    "5200",  # Neurosurgery
    "6000",  # Oncology
    "6100",  # Hematology
]

# Admission modes
ADMISSION_MODES = ["emergency", "planned", "transfer"]

# Error types to inject
ERROR_TYPES = [
    "missing_icd10_main",      # 5% - No main diagnosis → R001
    "invalid_date_range",      # 3% - Discharge before admission → R002
    "missing_jgp_code",        # 4% - Missing JGP code → R003
    "missing_procedures",      # 3% - No procedures → R004
    "invalid_admission_mode",  # 3% - Invalid admission mode → R005
    "missing_department",      # 3% - Missing department code → R006
    "zero_tariff",             # 2% - Zero or negative tariff → R007
]

ERROR_WEIGHTS = [0.22, 0.13, 0.17, 0.13, 0.13, 0.13, 0.09]  # Probability weights


# =============================================================================
# Helper Functions
# =============================================================================


def mask_pesel(pesel: str) -> str:
    """Mask PESEL with deterministic hash."""
    hash_val = hashlib.sha256(pesel.encode()).hexdigest()[:8]
    return f"PESEL_{hash_val}"


def generate_pesel(birth_date: datetime, is_female: bool) -> str:
    """Generate valid-format PESEL (not cryptographically valid checksum)."""
    year = birth_date.year
    month = birth_date.month
    day = birth_date.day

    # Century indicator
    if 1900 <= year <= 1999:
        month_code = month
    elif 2000 <= year <= 2099:
        month_code = month + 20
    else:
        month_code = month

    year_code = year % 100
    serial = random.randint(0, 999)
    gender_digit = random.choice([0, 2, 4, 6, 8]) if is_female else random.choice([1, 3, 5, 7, 9])

    pesel_base = f"{year_code:02d}{month_code:02d}{day:02d}{serial:03d}{gender_digit}"
    # Simplified checksum (last digit)
    checksum = sum(int(d) for d in pesel_base) % 10
    return pesel_base + str(checksum)


def generate_patient_id() -> str:
    """Generate masked patient ID."""
    return f"PAT_{uuid.uuid4().hex[:8].upper()}"


def generate_case_id(index: int) -> str:
    """Generate unique case ID."""
    return f"ENC{index:08d}"


def random_date_in_range(start: datetime, end: datetime) -> datetime:
    """Generate random datetime within range."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


# =============================================================================
# Claim Generator
# =============================================================================


def generate_valid_claim(index: int, reference_date: datetime) -> dict[str, Any]:
    """Generate a valid claim record with correct data."""
    # Patient demographics
    birth_year = random.randint(1940, 2005)
    birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
    is_female = random.choice([True, False])
    pesel = generate_pesel(birth_date, is_female)

    # Hospitalization dates
    admission_date = random_date_in_range(
        reference_date - timedelta(days=365),
        reference_date - timedelta(days=3)
    )
    los = random.randint(1, 14)  # Length of stay
    discharge_date = admission_date + timedelta(days=los)

    # Clinical data
    jgp_code = random.choice(list(JGP_CODES.keys()))
    jgp_info = JGP_CODES[jgp_code]

    # Main diagnosis related to JGP (simplified mapping)
    icd10_main = random.choice(ICD10_CODES)
    icd10_secondary = random.sample(ICD10_CODES, k=random.randint(0, 3))
    icd10_secondary = [c for c in icd10_secondary if c != icd10_main]

    # Procedures matching JGP
    procedures = jgp_info["procedures"].copy()
    if random.random() > 0.7:
        # Add extra procedure
        extra = random.choice([p for p in PROCEDURE_CODES if p not in procedures])
        procedures.append(extra)

    return {
        "case_id": generate_case_id(index),
        "patient_id": generate_patient_id(),
        "pesel_masked": mask_pesel(pesel),
        "birth_date": birth_date.strftime("%Y-%m-%d"),
        "gender": "F" if is_female else "M",
        "admission_date": admission_date.strftime("%Y-%m-%d"),
        "discharge_date": discharge_date.strftime("%Y-%m-%d"),
        "admission_mode": random.choice(ADMISSION_MODES),
        "department_code": random.choice(DEPARTMENT_CODES),
        "jgp_code": jgp_code,
        "jgp_name": jgp_info["name"],
        "tariff_value": jgp_info["tariff"],
        "icd10_main": icd10_main,
        "icd10_secondary": icd10_secondary,
        "procedures": procedures,
        "status": "new",
        "has_error": False,
        "error_type": None,
    }


def inject_error(claim: dict[str, Any], error_type: str) -> dict[str, Any]:
    """Inject a specific error type into a claim."""
    claim = claim.copy()
    claim["has_error"] = True
    claim["error_type"] = error_type

    if error_type == "missing_icd10_main":
        # R001: Missing main diagnosis
        claim["icd10_main"] = None

    elif error_type == "invalid_date_range":
        # R002: Discharge before admission
        admission = datetime.strptime(claim["admission_date"], "%Y-%m-%d")
        claim["discharge_date"] = (admission - timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d")

    elif error_type == "missing_jgp_code":
        # R003: Missing JGP code
        claim["jgp_code"] = None

    elif error_type == "missing_procedures":
        # R004: No procedures recorded
        claim["procedures"] = []

    elif error_type == "invalid_admission_mode":
        # R005: Invalid admission mode
        claim["admission_mode"] = random.choice(["unknown", "invalid", ""])

    elif error_type == "missing_department":
        # R006: Missing department code
        claim["department_code"] = None

    elif error_type == "zero_tariff":
        # R007: Zero or negative tariff
        claim["tariff_value"] = 0.0  # Will fail R007 rule

    return claim


def generate_batch(
    count: int,
    error_rate: float = 0.2,
    seed: int | None = None
) -> list[dict[str, Any]]:
    """
    Generate a batch of claims with controlled error injection.

    Args:
        count: Number of claims to generate
        error_rate: Fraction of claims with errors (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of claim dicts
    """
    if seed is not None:
        random.seed(seed)

    reference_date = datetime.now()
    claims = []
    error_count = int(count * error_rate)

    # Generate indices for error injection
    error_indices = set(random.sample(range(count), error_count))

    for i in range(count):
        claim = generate_valid_claim(i, reference_date)

        if i in error_indices:
            error_type = random.choices(ERROR_TYPES, weights=ERROR_WEIGHTS, k=1)[0]
            claim = inject_error(claim, error_type)

        claims.append(claim)

    return claims


# =============================================================================
# Output Functions
# =============================================================================


def claims_to_dataframe(claims: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert claims list to Polars DataFrame."""
    # Flatten list fields to JSON strings for Parquet compatibility
    records = []
    for claim in claims:
        record = claim.copy()
        record["icd10_secondary"] = ",".join(record["icd10_secondary"]) if record["icd10_secondary"] else ""
        record["procedures"] = ",".join(record["procedures"]) if record["procedures"] else ""
        records.append(record)

    return pl.DataFrame(records)


def save_parquet(df: pl.DataFrame, output_path: Path) -> None:
    """Save DataFrame to Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"✅ Saved {len(df)} records to {output_path}")


def save_csv(df: pl.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"✅ Saved {len(df)} records to {output_path}")


def print_statistics(claims: list[dict[str, Any]]) -> None:
    """Print generation statistics."""
    total = len(claims)
    errors = sum(1 for c in claims if c["has_error"])

    print("\n📊 Generation Statistics:")
    print(f"   Total records: {total:,}")
    print(f"   Records with errors: {errors:,} ({100 * errors / total:.1f}%)")
    print(f"   Valid records: {total - errors:,} ({100 * (total - errors) / total:.1f}%)")

    # Error breakdown
    error_counts: dict[str, int] = {}
    for claim in claims:
        if claim["error_type"]:
            error_counts[claim["error_type"]] = error_counts.get(claim["error_type"], 0) + 1

    if error_counts:
        print("\n   Error breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"   - {error_type}: {count} ({100 * count / total:.1f}%)")

    # JGP distribution (top 5)
    jgp_counts: dict[str, int] = {}
    for claim in claims:
        if claim["jgp_code"]:
            jgp_counts[claim["jgp_code"]] = jgp_counts.get(claim["jgp_code"], 0) + 1

    print("\n   Top 5 JGP groups:")
    for jgp, count in sorted(jgp_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"   - {jgp} ({JGP_CODES[jgp]['name']}): {count}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic NFZ claim data for Priqualis testing"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/raw/claims.parquet"),
        help="Output file path (supports .parquet and .csv)"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10000,
        help="Number of claims to generate"
    )
    parser.add_argument(
        "--error-rate", "-e",
        type=float,
        default=0.2,
        help="Fraction of claims with errors (0.0 to 1.0)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Output format"
    )

    args = parser.parse_args()

    print("🏥 Priqualis Synthetic Data Generator")
    print("=" * 40)
    print(f"   Count: {args.count:,}")
    print(f"   Error rate: {args.error_rate:.1%}")
    print(f"   Seed: {args.seed}")
    print(f"   Output: {args.output}")

    # Generate claims
    print("\n⏳ Generating claims...")
    claims = generate_batch(
        count=args.count,
        error_rate=args.error_rate,
        seed=args.seed
    )

    # Convert to DataFrame
    df = claims_to_dataframe(claims)

    # Save output
    if args.format in ("parquet", "both"):
        output_parquet = args.output.with_suffix(".parquet")
        save_parquet(df, output_parquet)

    if args.format in ("csv", "both"):
        output_csv = args.output.with_suffix(".csv")
        save_csv(df, output_csv)

    # Print statistics
    print_statistics(claims)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
