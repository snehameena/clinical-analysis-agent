"""
Integration tests for FastAPI endpoints.
All LLM/Tavily calls are mocked — no API keys needed.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.api.dependencies import run_store


@pytest.fixture
def client():
    """Create a test client with mocked graph."""
    with patch("src.api.dependencies.get_graph", return_value=MagicMock()):
        app = create_app()
        return TestClient(app)


@pytest.fixture(autouse=True)
def clear_run_store():
    """Clear the run store before each test."""
    run_store.clear()
    yield
    run_store.clear()


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data


class TestPipelineEndpoints:
    """Test pipeline CRUD endpoints."""

    def test_run_pipeline_returns_run_id(self, client):
        response = client.post("/api/v1/pipeline/run", json={
            "topic": "Telemedicine for diabetes",
        })
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"

    def test_get_status_for_existing_run(self, client):
        # Manually insert a state to avoid background task race
        from src.api.dependencies import run_store
        run_store["test-status-run"] = {
            "run_id": "test-status-run",
            "pipeline_status": "running",
            "current_agent": "coordinator",
            "quality_iteration": 0,
            "error_message": None,
        }

        status_resp = client.get("/api/v1/pipeline/test-status-run/status")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["run_id"] == "test-status-run"
        assert data["status"] == "running"
        assert data["current_agent"] == "coordinator"

    def test_get_status_404_for_unknown_run(self, client):
        response = client.get("/api/v1/pipeline/nonexistent-id/status")
        assert response.status_code == 404

    def test_get_report_400_when_not_completed(self, client):
        resp = client.post("/api/v1/pipeline/run", json={"topic": "Test"})
        run_id = resp.json()["run_id"]

        report_resp = client.get(f"/api/v1/pipeline/{run_id}/report")
        assert report_resp.status_code == 400

    def test_get_report_404_for_unknown_run(self, client):
        response = client.get("/api/v1/pipeline/nonexistent-id/report")
        assert response.status_code == 404

    def test_get_report_for_completed_run(self, client):
        # Manually insert a completed state
        run_id = "completed-test"
        run_store[run_id] = {
            "run_id": run_id,
            "pipeline_status": "completed",
            "report_markdown": "# Report\nContent here",
            "report_sections": {"Summary": "# Summary"},
            "citations": ["Ref 1"],
            "quality_score": 85.0,
            "quality_verdict": "pass",
            "report_word_count": 100,
        }

        response = client.get(f"/api/v1/pipeline/{run_id}/report")
        assert response.status_code == 200
        data = response.json()
        assert data["report_markdown"] == "# Report\nContent here"
        assert data["quality_score"] == 85.0

    def test_run_pipeline_with_all_fields(self, client):
        response = client.post("/api/v1/pipeline/run", json={
            "topic": "Telemedicine for diabetes",
            "scope_instructions": "Adults only",
            "target_audience": "researchers",
            "report_format": "full_report",
            "max_quality_iterations": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Pipeline run started"
