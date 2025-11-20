"""
Database Schemas for KrishiSetu

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercase of the class name (e.g., AppUser -> "appuser").

These schemas are used for validation before inserting into MongoDB.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class AppUser(BaseModel):
    name: str
    phone: str
    role: Literal["farmer", "worker", "buyer"]
    language: Optional[str] = Field(None, description="Preferred language code, e.g., kn, en")
    location: Optional[GeoPoint] = None
    trust_score: float = Field(0.0, ge=0, le=5)


class WorkerProfile(BaseModel):
    user_id: str
    skills: List[str] = Field(default_factory=list, description="e.g., harvesting, sowing")
    wage_expectation: float = Field(..., ge=0)
    availability_dates: List[str] = Field(default_factory=list, description="ISO dates the worker is available")
    max_distance_km: float = Field(15.0, ge=0)
    preferences: Optional[List[str]] = Field(default=None, description="e.g., full-day, half-day, seasonal")
    home_location: Optional[GeoPoint] = None


class FarmerJob(BaseModel):
    farmer_id: str
    task: str
    workers_needed: int = Field(..., ge=1)
    date_time: datetime
    wage_offer: float = Field(..., ge=0)
    crop_type: Optional[str] = None
    location: GeoPoint
    instructions: Optional[str] = None
    status: Literal["open", "matched", "confirmed", "completed", "cancelled"] = "open"
    matched_worker_ids: List[str] = Field(default_factory=list)
    confirmed_worker_ids: List[str] = Field(default_factory=list)


class JobAssignment(BaseModel):
    job_id: str
    worker_id: str
    status: Literal["pending", "confirmed", "declined", "completed", "cancelled"] = "pending"
    locked_at: Optional[datetime] = None


class BuyerRequest(BaseModel):
    buyer_id: str
    crops: List[str]
    quantity_kg: float = Field(..., ge=0)
    price_min: Optional[float] = Field(None, ge=0)
    price_max: Optional[float] = Field(None, ge=0)
    preference: Optional[Literal["pickup", "delivery"]] = None
    location: Optional[GeoPoint] = None
    verification_status: Literal["unverified", "verified", "blocked"] = "unverified"


class PriceTick(BaseModel):
    commodity: str
    region: str
    price: float
    timestamp: datetime
    volatility: Optional[float] = None
    source: Optional[str] = None


class PriceForecast(BaseModel):
    commodity: str
    region: str
    horizon_days: int
    forecast_curve: List[Dict[str, Any]]  # {"date": str, "price": float}
    trend: Literal["up", "down", "flat"]
    risk_alerts: List[str] = Field(default_factory=list)


class CropRecommendation(BaseModel):
    region: str
    season: Optional[str] = None
    recommended: List[Dict[str, Any]]  # items with crop, score, expected_price, workers_needed, rationale


class ChatMessage(BaseModel):
    from_id: str
    to_id: str
    content: str
    encrypted: bool = True
    sent_at: datetime


class Notification(BaseModel):
    user_id: str
    type: str
    title: str
    body: str
    data: Optional[Dict[str, Any]] = None


class SyncEvent(BaseModel):
    user_id: str
    device_id: str
    payload_type: str
    payload: Dict[str, Any]
    synced_at: datetime
