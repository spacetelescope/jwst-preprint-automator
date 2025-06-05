"""Data models for JWST preprint analysis."""

from pydantic import BaseModel, Field


class JWSTScienceLabelerModel(BaseModel):
    """Model for JWST science classification results."""
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason, MUST be exact substrings from the provided excerpts.")
    jwstscience: float = Field(..., description="Whether the paper contains JWST science, scored between 0 and 1")
    reason: str = Field(..., description="Justification for the given 'jwstscience' score based ONLY on the provided excerpts")


class JWSTDOILabelerModel(BaseModel):
    """Model for JWST DOI classification results."""
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason, MUST be exact substrings from the provided excerpts.")
    jwstdoi: float = Field(..., description="Whether the JWST data is accompanied by a DOI (specifically 10.17909 prefix)")
    reason: str = Field(..., description="Justification for the given 'jwstdoi' score based ONLY on the provided excerpts")