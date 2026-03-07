"""
Environment loading helper.

Supports `.env.local` for developer-specific secrets while keeping `.env` as an optional fallback.

Precedence:
1) Existing process environment (never overridden)
2) `.env.local`
3) `.env`
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Load developer-local secrets first so `.env` cannot override them.
    load_dotenv(repo_root / ".env.local", override=False)
    load_dotenv(repo_root / ".env", override=False)

