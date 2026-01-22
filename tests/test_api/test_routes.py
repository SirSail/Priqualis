"""Tests for API routes."""

from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_validate_single(client: TestClient, valid_claim_payload):
    """Test validating a single claim."""
    response = client.post("/api/v1/validate/single", json=valid_claim_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == valid_claim_payload["case_id"]
    assert data["violation_count"] == 0

def test_validate_batch(client: TestClient, valid_claim_payload, invalid_claim_payload):
    """Test validating a batch of claims."""
    batch = {"claims": [valid_claim_payload, invalid_claim_payload]}
    response = client.post("/api/v1/validate/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert data["violation_count"] > 0
    
    # Check that invalid claim has violations
    violations = data["results"]
    r002 = next((v for v in violations if v["rule_id"] == "R002"), None)
    assert r002 is not None
    assert r002["case_id"] == invalid_claim_payload["case_id"]

def test_autofix_generation(client: TestClient, invalid_claim_payload):
    """Test autofix generation endpoint."""
    # R002 (date error) should have autofix
    batch = {"claims": [invalid_claim_payload]}
    response = client.post("/api/v1/autofix/generate", json=batch)
    
    assert response.status_code == 200
    patches = response.json()
    assert len(patches) > 0
    
    patch = patches[0]
    assert patch["rule_id"] == "R002"
    assert patch["changes"][0]["op"] == "set"
    assert patch["changes"][0]["field"] == "discharge_date"

def test_search_similar(client: TestClient):
    """Test search endpoint (mocked data)."""
    # Note: Search depends on indexed data. 
    # Logic is likely to return empty list if index not built, but should not crash.
    query = {
        "case_id": "QUERY_001",
        "jgp_code": "A01", 
        "icd10_main": "J18.9"
    }
    response = client.post("/api/v1/similar/search", json=query)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
