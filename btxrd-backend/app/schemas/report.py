"""Pydantic schemas for report generation."""

from pydantic import BaseModel


class ReportRequest(BaseModel):
    image_id: str
    analysis: dict  # full inference result
    case_id: str | None = None
    patient_name: str | None = None
    patient_age: int | None = None
    clinical_indication: str | None = None


class ReportResponse(BaseModel):
    report: str
    pdf_url: str | None = None
