"""
HTTP client wrapping all backend API endpoints.
"""

import os
import httpx
from typing import Optional, Dict, Any

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class PipelineAPIClient:
    """Client for the Healthcare Intelligence Pipeline API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or BACKEND_URL).rstrip("/")

    def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        resp = httpx.get(f"{self.base_url}/api/v1/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def start_run(
        self,
        topic: str,
        scope_instructions: str = "",
        target_audience: str = "clinical practitioners",
        report_format: str = "clinical_brief",
        max_quality_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Launch a new pipeline run."""
        payload = {
            "topic": topic,
            "scope_instructions": scope_instructions,
            "target_audience": target_audience,
            "report_format": report_format,
            "max_quality_iterations": max_quality_iterations,
        }
        resp = httpx.post(
            f"{self.base_url}/api/v1/pipeline/run",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Poll the status of a pipeline run."""
        resp = httpx.get(
            f"{self.base_url}/api/v1/pipeline/{run_id}/status",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_report(self, run_id: str) -> Dict[str, Any]:
        """Retrieve the final report for a completed run."""
        resp = httpx.get(
            f"{self.base_url}/api/v1/pipeline/{run_id}/report",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
