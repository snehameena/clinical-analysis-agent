"""
Tavily search integration with healthcare domain filtering.
Provides web search capability with medical domain prioritization.
"""

import asyncio
import os
import json
from typing import Optional, List, Dict, Any
from tavily import TavilyClient
from src.state.schema import Source


class HealthcareTavilySearch:
    """Healthcare-specific Tavily search client"""

    # Healthcare domains to prioritize
    HEALTHCARE_DOMAINS = [
        "pubmed.ncbi.nlm.nih.gov",
        "thelancet.com",
        "jamanetwork.com",
        "bmj.com",
        "cochranelibrary.com",
        "who.int",
        "cdc.gov",
        "nih.gov",
        "nejm.org",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily search client"""
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not set in environment")
        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = int(os.getenv("TAVILY_MAX_RESULTS_PER_QUERY", 5))
        self.min_relevance = float(os.getenv("TAVILY_MIN_RELEVANCE_SCORE", 0.6))

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Search for healthcare evidence using Tavily.

        Args:
            query: Search query string

        Returns:
            Dict with 'results' containing search results
        """
        try:
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                search_depth="advanced",
                max_results=self.max_results,
                include_domains=self.HEALTHCARE_DOMAINS,
            )
            return response
        except Exception as e:
            return {"results": [], "error": str(e)}

    def parse_results(self, raw_results: Dict[str, Any]) -> List[Source]:
        """
        Parse Tavily results into Source objects.

        Args:
            raw_results: Raw Tavily API response

        Returns:
            List of Source objects with metadata
        """
        sources = []

        for result in raw_results.get("results", []):
            # Extract relevance score
            score = result.get("score", 0.5)

            # Filter by minimum relevance threshold
            if score < self.min_relevance:
                continue

            # Classify evidence level
            evidence_level = self._classify_evidence_level(result)

            source = Source(
                url=result.get("url", ""),
                title=result.get("title", ""),
                snippet=result.get("content", "")[:500],  # Truncate to 500 chars
                relevance_score=score,
                evidence_level=evidence_level,
                publication_date=result.get("publication_date"),
                authors=result.get("authors", []),
            )
            sources.append(source)

        return sources

    @staticmethod
    def _classify_evidence_level(result: Dict[str, Any]) -> str:
        """
        Classify evidence level based on source metadata and content.

        Args:
            result: Tavily search result

        Returns:
            Evidence level classification string
        """
        title_lower = result.get("title", "").lower()
        content_lower = result.get("content", "").lower()
        url_lower = result.get("url", "").lower()

        # Systematic Review
        if any(term in title_lower or term in content_lower for term in
               ["systematic review", "meta-analysis", "cochrane"]):
            return "Systematic Review"

        # RCT
        if any(term in title_lower or term in content_lower for term in
               ["randomized controlled trial", "rct", "randomized"]):
            return "Randomized Controlled Trial"

        # Cohort Study
        if any(term in title_lower or term in content_lower for term in
               ["cohort study", "prospective study", "longitudinal"]):
            return "Cohort Study"

        # Case Report
        if any(term in title_lower or term in content_lower for term in
               ["case report", "case series", "clinical case"]):
            return "Case Report"

        # Default to Opinion if from healthcare domain
        if any(domain in url_lower for domain in HealthcareTavilySearch.HEALTHCARE_DOMAINS):
            return "Expert Opinion"

        return "Expert Opinion"


def deduplicate_sources(sources: List[Source], min_relevance: float = 0.6) -> List[Source]:
    """
    Deduplicate sources by URL and filter by relevance.

    Args:
        sources: List of Source objects
        min_relevance: Minimum relevance score to include (0-1)

    Returns:
        Deduplicated list of sources sorted by relevance
    """
    seen_urls = {}

    for source in sources:
        url = source.get("url", "")
        relevance = source.get("relevance_score", 0)

        # Keep the highest relevance score for each URL
        if url not in seen_urls or relevance > seen_urls[url]["relevance_score"]:
            seen_urls[url] = source

    # Filter by minimum relevance and sort by relevance descending
    deduplicated = [
        source for source in seen_urls.values()
        if source.get("relevance_score", 0) >= min_relevance
    ]

    return sorted(deduplicated, key=lambda x: x.get("relevance_score", 0), reverse=True)
