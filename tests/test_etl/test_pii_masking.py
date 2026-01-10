"""Tests for PII masking."""

import pytest

from priqualis.etl.pii_masking import (
    PIIMasker,
    mask_pesel,
    mask_name,
    mask_address,
    mask_phone,
    MaskingRule,
)


class TestMaskingFunctions:
    """Tests for individual masking functions."""

    def test_mask_pesel_valid(self):
        """Test PESEL masking with valid 11-digit input."""
        pesel = "92010112345"
        masked = mask_pesel(pesel)

        assert masked != pesel
        assert masked.startswith("PESEL_")
        assert len(masked) == 14  # PESEL_ + 8 hex chars

    def test_mask_pesel_deterministic(self):
        """Test that same PESEL always gives same hash."""
        pesel = "92010112345"

        masked1 = mask_pesel(pesel)
        masked2 = mask_pesel(pesel)

        assert masked1 == masked2

    def test_mask_pesel_different_inputs(self):
        """Test that different PESELs give different hashes."""
        masked1 = mask_pesel("92010112345")
        masked2 = mask_pesel("85050512345")

        assert masked1 != masked2

    def test_mask_pesel_invalid_length(self):
        """Test masking invalid PESEL returns original."""
        invalid = "12345"  # Too short
        masked = mask_pesel(invalid)

        # Returns unchanged when length != 11
        assert masked == invalid

    def test_mask_pesel_empty(self):
        """Test masking empty string."""
        assert mask_pesel("") == ""
        assert mask_pesel(None) is None

    def test_mask_name_valid(self):
        """Test name masking."""
        name = "Jan Kowalski"
        masked = mask_name(name)

        assert masked != name
        assert masked.startswith("PAT_")

    def test_mask_name_empty(self):
        """Test masking None/empty name."""
        assert mask_name(None) is None
        assert mask_name("") == ""

    def test_mask_address(self):
        """Test address masking returns placeholder."""
        address = "ul. Główna 15, 00-001 Warszawa"
        masked = mask_address(address)

        assert masked == "[MASKED_ADDRESS]"

    def test_mask_phone(self):
        """Test phone masking returns placeholder."""
        phone = "+48 123 456 789"
        masked = mask_phone(phone)

        assert masked == "[MASKED_PHONE]"


class TestPIIMasker:
    """Tests for PIIMasker class."""

    @pytest.fixture
    def masker(self) -> PIIMasker:
        return PIIMasker()

    def test_mask_dict_basic(self, masker: PIIMasker):
        """Test masking a dictionary with PII fields."""
        record = {
            "pesel": "92010112345",
            "patient_name": "Jan Kowalski",
            "case_id": "ENC123",  # Protected field
            "icd10_main": "J18.9",  # Protected field
        }

        masked, count = masker.mask_dict(record)

        # PII fields should be masked
        assert masked["pesel"] != record["pesel"]
        assert masked["patient_name"] != record["patient_name"]

        # Protected fields unchanged
        assert masked["case_id"] == record["case_id"]
        assert masked["icd10_main"] == record["icd10_main"]

    def test_mask_dict_missing_fields(self, masker: PIIMasker):
        """Test masking dict without PII fields."""
        record = {"case_id": "ENC123", "jgp_code": "A01"}

        masked, count = masker.mask_dict(record)

        assert count == 0
        assert masked == record


class TestMaskingRule:
    """Tests for MaskingRule dataclass."""

    def test_masking_rule_immutable(self):
        """Test MaskingRule is frozen (immutable)."""
        rule = MaskingRule(
            field_name="pesel",
            mask_fn=mask_pesel,
            description="PESEL masking",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            rule.field_name = "other"
