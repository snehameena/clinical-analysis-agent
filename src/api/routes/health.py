"""
Health check endpoint.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/v1/health")
async def health_check():
    """Return service health status."""
    return {"status": "healthy", "service": "healthcare-intelligence-pipeline"}
