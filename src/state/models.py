"""
Pydantic models for validating/normalizing agent outputs before writing into PipelineState.

Goal: keep PipelineState type-stable even when LLM outputs vary in shape (string vs list, dict vs list, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _norm_evidence_level(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s:
        return ""
    key = s.lower()
    mapping = {
        "rct": "Randomized Controlled Trial",
        "randomised controlled trial": "Randomized Controlled Trial",
        "randomized trial": "Randomized Controlled Trial",
        "randomized controlled trial": "Randomized Controlled Trial",
        "systematic review": "Systematic Review",
        "meta-analysis": "Systematic Review",
        "meta analysis": "Systematic Review",
        "cohort": "Cohort Study",
        "cohort study": "Cohort Study",
        "expert opinion": "Expert Opinion",
        "opinion": "Expert Opinion",
        "case report": "Case Report",
        "case series": "Case Report",
    }
    return mapping.get(key, s)


class CoordinatorOutput(BaseModel):
    research_queries: List[str] = Field(default_factory=list)
    scope_boundaries: Dict[str, List[str]] = Field(default_factory=lambda: {"in_scope": [], "out_of_scope": []})
    priority_subtopics: List[str] = Field(default_factory=list)

    @field_validator("research_queries", mode="before")
    @classmethod
    def _coerce_queries(cls, v: Any) -> List[str]:
        items = _as_list(v)
        out: List[str] = []
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    @field_validator("priority_subtopics", mode="before")
    @classmethod
    def _coerce_subtopics(cls, v: Any) -> List[str]:
        return cls._coerce_queries(v)

    @field_validator("scope_boundaries", mode="before")
    @classmethod
    def _coerce_scope(cls, v: Any) -> Dict[str, List[str]]:
        if v is None:
            return {"in_scope": [], "out_of_scope": []}
        if not isinstance(v, dict):
            return {"in_scope": [], "out_of_scope": []}
        in_scope = _as_list(v.get("in_scope"))
        out_scope = _as_list(v.get("out_of_scope"))
        def _clean(xs: List[Any]) -> List[str]:
            out: List[str] = []
            for x in xs:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        return {"in_scope": _clean(in_scope), "out_of_scope": _clean(out_scope)}


class ClinicalClaimModel(BaseModel):
    claim: str = ""
    evidence_level: str = ""
    source_urls: List[str] = Field(default_factory=list)
    effect_size: Optional[str] = None
    confidence_interval: Optional[str] = None
    p_value: Optional[str] = None

    @field_validator("source_urls", mode="before")
    @classmethod
    def _coerce_urls(cls, v: Any) -> List[str]:
        items = _as_list(v)
        out: List[str] = []
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    @field_validator("evidence_level", mode="before")
    @classmethod
    def _coerce_evidence(cls, v: Any) -> str:
        return _norm_evidence_level(v)


class AnalysisOutput(BaseModel):
    clinical_claims: List[ClinicalClaimModel] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    statistical_findings: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_narrative: str = ""

    @field_validator("clinical_claims", mode="before")
    @classmethod
    def _coerce_claims(cls, v: Any) -> List[Any]:
        return _as_list(v)

    @field_validator("evidence_gaps", "contradictions", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> List[str]:
        items = _as_list(v)
        out: List[str] = []
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    @field_validator("statistical_findings", mode="before")
    @classmethod
    def _coerce_stats(cls, v: Any) -> List[Dict[str, Any]]:
        items = _as_list(v)
        out: List[Dict[str, Any]] = []
        for x in items:
            if isinstance(x, dict):
                out.append(x)
        return out


class WritingOutput(BaseModel):
    report_sections: Dict[str, str] = Field(default_factory=dict)
    report_markdown: str = ""
    citations: List[str] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v: Any) -> List[str]:
        items = _as_list(v)
        out: List[str] = []
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    @field_validator("report_sections", mode="before")
    @classmethod
    def _coerce_sections(cls, v: Any) -> Dict[str, str]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            return {}
        out: Dict[str, str] = {}
        for k, val in v.items():
            key = str(k).strip()
            if not key:
                continue
            out[key] = "" if val is None else str(val)
        return out


class QualityIssueModel(BaseModel):
    severity: str = "minor"
    section: str = ""
    description: str = ""
    recommendation: str = ""


class QualityOutput(BaseModel):
    quality_score: float = 0.0
    quality_verdict: Literal["pass", "revise", "reject"] = "reject"
    quality_issues: List[QualityIssueModel] = Field(default_factory=list)
    revision_instructions: str = ""

    @field_validator("quality_verdict", mode="before")
    @classmethod
    def _coerce_verdict(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        if s in ("pass", "revise", "reject"):
            return s
        return "reject"

    @field_validator("quality_issues", mode="before")
    @classmethod
    def _coerce_issues(cls, v: Any) -> List[Any]:
        return _as_list(v)

    @field_validator("revision_instructions", mode="before")
    @classmethod
    def _coerce_revision_instructions(cls, v: Any) -> str:
        """
        LLMs sometimes return revision_instructions as a dict like:
          {"Executive Summary": ["...", "..."], "Evidence": ["..."]}
        Normalize to a single string that WritingAgent can consume.
        """
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts: List[str] = []
            for x in v:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    parts.append(s)
            return "\n".join(parts)
        if isinstance(v, dict):
            lines: List[str] = []
            for section, items in v.items():
                section_name = str(section).strip()
                if not section_name:
                    section_name = "Section"
                lines.append(f"{section_name}:")
                for item in _as_list(items):
                    if item is None:
                        continue
                    s = str(item).strip()
                    if s:
                        lines.append(f"- {s}")
                lines.append("")  # spacer
            return "\n".join(lines).strip()
        return str(v)
