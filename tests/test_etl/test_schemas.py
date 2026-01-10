"""Tests for ETL schemas."""

import pytest
from pydantic import ValidationError

from priqualis.etl.schemas import (
    ClaimRecord,
    ClaimBatch,
    AdmissionMode,
    ClaimStatus,
    Gender,
)


class TestClaimRecord:
    """Tests for ClaimRecord model."""

    def test_valid_claim(self, sample_claim_dict: dict):
        """Test creating a valid claim."""
        claim = ClaimRecord(**sample_claim_dict)

        assert claim.case_id == "ENC12345"
        assert claim.icd10_main == "J18.9"
        assert claim.admission_mode == AdmissionMode.EMERGENCY
        assert len(claim.procedures) == 2

    def test_missing_required_field(self, sample_claim_dict: dict):
        """Test that missing required fields raise error."""
        del sample_claim_dict["case_id"]

        with pytest.raises(ValidationError) as exc_info:
            ClaimRecord(**sample_claim_dict)

        assert "case_id" in str(exc_info.value)

    def test_invalid_admission_mode_stored_as_string(self, sample_claim_dict: dict):
        """Test that invalid admission mode is stored as string (for error detection)."""
        sample_claim_dict["admission_mode"] = "invalid_mode"

        # Should NOT raise - we allow invalid modes for rule detection
        claim = ClaimRecord(**sample_claim_dict)
        assert claim.admission_mode == "invalid_mode"

    def test_optional_fields_none(self, sample_claim_dict: dict):
        """Test that optional fields can be None."""
        sample_claim_dict["icd10_main"] = None
        sample_claim_dict["icd10_secondary"] = None

        claim = ClaimRecord(**sample_claim_dict)

        assert claim.icd10_main is None
        # icd10_secondary defaults to empty list
        assert claim.icd10_secondary == []

    def test_empty_procedures_list(self, sample_claim_dict: dict):
        """Test empty procedures list is valid."""
        sample_claim_dict["procedures"] = []

        claim = ClaimRecord(**sample_claim_dict)

        assert claim.procedures == []

    def test_date_parsing(self, sample_claim_dict: dict):
        """Test date string parsing."""
        claim = ClaimRecord(**sample_claim_dict)

        assert claim.admission_date.year == 2024
        assert claim.admission_date.month == 1
        assert claim.admission_date.day == 15

    def test_tariff_zero_valid(self, sample_claim_dict: dict):
        """Test that zero tariff is valid (ge=0.0)."""
        sample_claim_dict["tariff_value"] = 0.0

        claim = ClaimRecord(**sample_claim_dict)
        assert claim.tariff_value == 0.0

    def test_tariff_negative_raises(self, sample_claim_dict: dict):
        """Test that negative tariff raises error."""
        sample_claim_dict["tariff_value"] = -100.0

        with pytest.raises(ValidationError):
            ClaimRecord(**sample_claim_dict)

    def test_length_of_stay_property(self, sample_claim_dict: dict):
        """Test length_of_stay calculation."""
        claim = ClaimRecord(**sample_claim_dict)

        # 2024-01-15 to 2024-01-18 = 3 days
        assert claim.length_of_stay == 3

    def test_is_valid_date_range_property(self, sample_claim_dict: dict):
        """Test is_valid_date_range property."""
        claim = ClaimRecord(**sample_claim_dict)
        assert claim.is_valid_date_range is True

        # Invalid: discharge before admission
        sample_claim_dict["discharge_date"] = "2024-01-10"
        invalid_claim = ClaimRecord(**sample_claim_dict)
        assert invalid_claim.is_valid_date_range is False


class TestClaimBatch:
    """Tests for ClaimBatch model."""

    def test_valid_batch(self, sample_claim_batch: ClaimBatch):
        """Test batch properties."""
        assert sample_claim_batch.count >= 1

    def test_empty_batch(self):
        """Test empty batch is valid."""
        batch = ClaimBatch(records=[])

        assert batch.count == 0
        assert batch.error_count == 0
        assert batch.valid_count == 0

    def test_get_by_case_id(self, sample_claim_batch: ClaimBatch):
        """Test finding record by case_id."""
        record = sample_claim_batch.get_by_case_id("ENC12345")

        assert record is not None
        assert record.case_id == "ENC12345"

    def test_get_by_case_id_not_found(self, sample_claim_batch: ClaimBatch):
        """Test get_by_case_id returns None for missing ID."""
        record = sample_claim_batch.get_by_case_id("NONEXISTENT")

        assert record is None

    def test_filter_by_status(self, sample_claim: ClaimRecord):
        """Test filtering by status."""
        batch = ClaimBatch(records=[sample_claim])

        new_records = batch.filter_by_status(ClaimStatus.NEW)
        assert len(new_records) == 1

        validated_records = batch.filter_by_status(ClaimStatus.VALIDATED)
        assert len(validated_records) == 0
