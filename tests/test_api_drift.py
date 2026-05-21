"""Tests for the /internal/drift-score push endpoint (phase 5.4)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


class TestDriftEndpoint:
    def test_requires_api_key(self, client: TestClient) -> None:
        r = client.post(
            "/internal/drift-score",
            json={"feature_set": "hog", "score": 0.1},
        )
        assert r.status_code == 401

    def test_valid_payload_returns_204_and_updates_metric(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        r = client.post(
            "/internal/drift-score",
            json={"feature_set": "hog", "score": 0.42},
            headers=api_key_header,
        )
        assert r.status_code == 204
        metrics_body = client.get("/metrics").text
        assert 'metrovision_drift_score{feature_set="hog"} 0.42' in metrics_body

    def test_score_out_of_range_returns_422(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        r = client.post(
            "/internal/drift-score",
            json={"feature_set": "hog", "score": 1.5},
            headers=api_key_header,
        )
        assert r.status_code == 422

    def test_invalid_feature_set_name_returns_422(
        self, client: TestClient, api_key_header: dict[str, str]
    ) -> None:
        r = client.post(
            "/internal/drift-score",
            json={"feature_set": "has spaces!", "score": 0.1},
            headers=api_key_header,
        )
        assert r.status_code == 422
