"""
Text utilities for the pipeline.
Provides token counting, citation formatting, and text processing.
"""

import re
from typing import List, Dict, Optional, Tuple
from src.state.schema import Source


def estimate_token_count(text: str, model: str = "claude-3-5-sonnet-20241022") -> int:
    """
    Estimate token count for text (without using expensive tokenizer).

    Uses rule of thumb: ~4 chars = 1 token for English text.
    This is a rough approximation; for exact counts use Anthropic's API.

    Args:
        text: Text to count tokens for
        model: Model name (for future use with exact tokenizers)

    Returns:
        Estimated token count
    """
    # Rule of thumb: 1 token ≈ 4 characters for English
    # Add 10% overhead for special tokens
    return int(len(text) / 4 * 1.1)


def format_mla_citation(source: Source) -> str:
    """
    Format source as MLA citation.

    Args:
        source: Source object

    Returns:
        MLA formatted citation string
    """
    title = source.get("title", "Unknown Title")
    authors = source.get("authors", [])
    publication_date = source.get("publication_date", "")
    url = source.get("url", "")

    # Format authors
    if authors:
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]}, and {authors[1]}"
        else:
            author_str = f"{authors[0]}, et al."
    else:
        author_str = "Author Unknown"

    # Build citation
    citation = f"{author_str}. \"{title}.\" "

    # Add URL
    if url:
        citation += f"Accessed from {url}"

    if publication_date:
        citation += f" ({publication_date})"

    return citation


def format_harvard_citation(source: Source) -> str:
    """
    Format source as Harvard citation.

    Args:
        source: Source object

    Returns:
        Harvard formatted citation string
    """
    title = source.get("title", "Unknown Title")
    authors = source.get("authors", [])
    publication_date = source.get("publication_date", "")
    url = source.get("url", "")

    # Format authors
    if authors:
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        else:
            author_str = f"{authors[0]} et al"
    else:
        author_str = "Author Unknown"

    # Build citation
    citation = f"{author_str}"
    if publication_date:
        year = publication_date.split("-")[0] if publication_date else ""
        if year:
            citation += f" ({year})"

    citation += f" {title}"

    if url:
        citation += f" Available at: {url}"

    return citation


def format_citation(source: Source, style: str = "harvard") -> str:
    """
    Format source citation in specified style.

    Args:
        source: Source object
        style: Citation style ("harvard", "mla", or "url")

    Returns:
        Formatted citation
    """
    if style.lower() == "mla":
        return format_mla_citation(source)
    elif style.lower() == "harvard":
        return format_harvard_citation(source)
    else:
        # Default to URL only
        return source.get("url", "")


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from text.

    Args:
        text: Text to extract URLs from

    Returns:
        List of unique URLs found
    """
    url_pattern = r"https?://[^\s\)\"]+|www\.[^\s\)\"]+\.[^\s\)]+"
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates


def create_markdown_link(text: str, url: str) -> str:
    """
    Create markdown link.

    Args:
        text: Link text
        url: Link URL

    Returns:
        Markdown link string
    """
    return f"[{text}]({url})"


def sanitize_markdown(text: str) -> str:
    """
    Sanitize markdown by escaping dangerous characters.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for markdown rendering
    """
    # Escape common markdown special characters in headings context
    # but preserve intentional markdown formatting
    return text.replace("\\", "\\\\")


def truncate_text(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum character count.

    Args:
        text: Text to truncate
        max_chars: Maximum character length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Simple sentence split on . ! ? followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Calculate estimated reading time in minutes.

    Args:
        text: Text to analyze
        words_per_minute: Average reading speed (default 200)

    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, word_count // words_per_minute)


def create_evidence_badge(evidence_level: str) -> str:
    """
    Create markdown badge for evidence level.

    Args:
        evidence_level: Evidence level string

    Returns:
        Markdown badge string
    """
    badge_map = {
        "Systematic Review": "🟢 Systematic Review",
        "Randomized Controlled Trial": "🟢 RCT",
        "Cohort Study": "🟡 Cohort Study",
        "Case Report": "🟠 Case Report",
        "Expert Opinion": "🔵 Expert Opinion",
    }
    return badge_map.get(evidence_level, f"⚪ {evidence_level}")


def format_relevance_score(score: float) -> str:
    """
    Format relevance score as human-readable string.

    Args:
        score: Relevance score (0-1)

    Returns:
        Human-readable relevance description
    """
    if score >= 0.9:
        return "Highly Relevant"
    elif score >= 0.7:
        return "Very Relevant"
    elif score >= 0.6:
        return "Relevant"
    else:
        return "Low Relevance"


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Text to count

    Returns:
        Word count
    """
    return len(text.split())


def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords from text (simple TF-based approach).

    Args:
        text: Text to analyze
        num_keywords: Number of keywords to extract

    Returns:
        List of keywords
    """
    # Simple implementation: split by non-alphanumeric, remove small words
    words = re.findall(r'\b\w{4,}\b', text.lower())

    # Count word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:num_keywords]]
