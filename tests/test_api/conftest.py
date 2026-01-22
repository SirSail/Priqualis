"""Fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)

@pytest.fixture
def valid_claim_payload():
    """Valid claim payload for testing."""
    return {
        "case_id": "TEST_001",
        "jgp_code": "A01",
        "icd10_main": "J18.9",
        "icd10_secondary": ["I10"],
        "procedures": ["89.00"],
        "admission_date": "2024-01-01",
        "discharge_date": "2024-01-05", # Valid > admission
        "department_code": "4000",
        "admission_mode": "planned",
        "tariff_value": 1500.0,
    }

@pytest.fixture
def invalid_claim_payload():
    """Invalid claim payload (R002 violation)."""
    return {
        "case_id": "TEST_002",
        "jgp_code": "A01",
        "icd10_main": "J18.9",
        "admission_date": "2024-01-05",
        "discharge_date": "2024-01-01", # Invalid < admission
        "department_code": "4000",
        "admission_mode": "planned",
    }
