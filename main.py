import os
import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import (
    AppUser,
    WorkerProfile,
    FarmerJob,
    JobAssignment,
    BuyerRequest,
    PriceTick,
    PriceForecast,
    CropRecommendation,
    ChatMessage,
    Notification,
    SyncEvent,
    GeoPoint,
)

app = FastAPI(title="KrishiSetu Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Utilities
# -----------------------

def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        if isinstance(v, dict):
            d[k] = serialize_doc(v)
        if isinstance(v, list):
            d[k] = [serialize_doc(x) if isinstance(x, dict) else x for x in v]
    return d


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def detect_language(text: str) -> str:
    # Extremely naive language detection: Kannada unicode range check
    if any("\u0C80" <= ch <= "\u0CFF" for ch in text):
        return "kn"
    return "en"


def simple_translate_to_en(text: str, lang: str) -> str:
    # Placeholder translation. In production, integrate a translation API.
    return text


def parse_int_in_text(text: str) -> Optional[int]:
    import re
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def parse_task(text: str) -> Optional[str]:
    keywords = [
        "harvesting",
        "harvest",
        "planting",
        "sowing",
        "weeding",
        "loading",
        "transport",
    ]
    t = text.lower()
    for k in keywords:
        if k in t:
            return "harvesting" if k.startswith("harvest") else k
    return None


def parse_when(text: str) -> datetime:
    t = text.lower()
    now = datetime.utcnow()
    if "tomorrow" in t:
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    if "today" in t:
        return now.replace(hour=9, minute=0, second=0, microsecond=0)
    # Fallback: next day morning
    return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)


# -----------------------
# 0. Health & Schema
# -----------------------

@app.get("/")
def read_root():
    return {"message": "KrishiSetu Backend Running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected & Working"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:20]
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:80]}"
    return response


@app.get("/schema")
def get_schema():
    from schemas import (
        AppUser,
        WorkerProfile,
        FarmerJob,
        JobAssignment,
        BuyerRequest,
        PriceTick,
        PriceForecast,
        CropRecommendation,
        ChatMessage,
        Notification,
        SyncEvent,
    )

    def model_to_desc(model):
        return {
            "name": model.__name__,
            "fields": list(model.model_fields.keys()),
        }

    models = [
        AppUser,
        WorkerProfile,
        FarmerJob,
        JobAssignment,
        BuyerRequest,
        PriceTick,
        PriceForecast,
        CropRecommendation,
        ChatMessage,
        Notification,
        SyncEvent,
    ]
    return {"models": [model_to_desc(m) for m in models]}


# -----------------------
# 1. Identification & Language Processing
# -----------------------

class LoginInput(BaseModel):
    name: str
    phone: str
    role: str  # farmer | worker | buyer
    language: Optional[str] = None
    location: Optional[GeoPoint] = None


@app.post("/auth/login")
def login(payload: LoginInput):
    if payload.role not in {"farmer", "worker", "buyer"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    user = AppUser(
        name=payload.name,
        phone=payload.phone,
        role=payload.role,
        language=payload.language or "en",
        location=payload.location,
        trust_score=0.0,
    )
    coll = db["appuser"]
    coll.update_one({"phone": user.phone}, {"$set": user.model_dump()}, upsert=True)
    doc = coll.find_one({"phone": user.phone})
    return {
        "user": serialize_doc(doc),
        "token": f"token-{doc['_id']}",  # placeholder token
    }


class ASRInput(BaseModel):
    audio_b64: str
    language: Optional[str] = None


@app.post("/asr/transcribe")
def transcribe(input: ASRInput):
    # Placeholder ASR; integrate with a real ASR service in production
    detected_lang = input.language or "en"
    text = "Transcription unavailable in demo. Please use text input."
    return {"language": detected_lang, "text": text}


class NLPInput(BaseModel):
    text: str
    location: Optional[GeoPoint] = None


@app.post("/nlp/parse")
def nlp_parse(inp: NLPInput):
    lang = detect_language(inp.text)
    text_en = inp.text if lang == "en" else simple_translate_to_en(inp.text, lang)
    task = parse_task(text_en) or "general"
    workers = parse_int_in_text(text_en) or 1
    when = parse_when(text_en)
    return {
        "language": lang,
        "task": task,
        "workers": workers,
        "date_time": when.isoformat(),
        "location": inp.location.model_dump() if inp.location else None,
    }


# -----------------------
# 2. Job Creation & Worker Matching
# -----------------------

class CreateJobInput(BaseModel):
    farmer_id: str
    task: str
    workers_needed: int
    date_time: datetime
    wage_offer: float
    crop_type: Optional[str] = None
    location: GeoPoint
    instructions: Optional[str] = None


@app.post("/jobs")
def create_job(inp: CreateJobInput):
    job = FarmerJob(**inp.model_dump())
    job_id = create_document("farmerjob", job)
    matches = match_job(job_id)
    return {"job_id": job_id, "matches": matches}


def match_job(job_id: str) -> List[Dict[str, Any]]:
    job_doc = db["farmerjob"].find_one({"_id": {'$eq': db["farmerjob"]._Database__client.codec_options.document_class().from_oid(job_id) if False else None}})
    # Fallback: fetch by string then re-fetch by _id conversion manually
    from bson import ObjectId
    try:
        job_doc = db["farmerjob"].find_one({"_id": ObjectId(job_id)})
    except Exception:
        job_doc = db["farmerjob"].find_one({"_id": job_id})
    if not job_doc:
        return []

    job = serialize_doc(job_doc)
    task = job.get("task", "").lower()
    lat = job["location"]["lat"]
    lon = job["location"]["lon"]
    offer = float(job.get("wage_offer", 0))
    need_date = job["date_time"][:10]

    workers = list(db["workerprofile"].find({}))
    ranked: List[Dict[str, Any]] = []
    for w in workers:
        w_ser = serialize_doc(w)
        skills = [s.lower() for s in w_ser.get("skills", [])]
        if task and task not in skills:
            continue
        if w_ser.get("wage_expectation", 0) > offer:
            continue
        # availability
        if w_ser.get("availability_dates") and need_date not in w_ser["availability_dates"]:
            continue
        # distance
        home = w_ser.get("home_location") or {}
        if home:
            d_km = haversine_km(lat, lon, home.get("lat", lat), home.get("lon", lon))
            if d_km > float(w_ser.get("max_distance_km", 9999)):
                continue
        else:
            d_km = 9999
        # reliability from user trust_score
        user_doc = db["appuser"].find_one({"_id": w.get("user_id")})
        if not user_doc:
            user_doc = db["appuser"].find_one({"_id": w.get("user_id", None)})
        trust = 3.0
        if user_doc:
            user_ser = serialize_doc(user_doc)
            trust = float(user_ser.get("trust_score", 3.0))
        experience = len(skills)
        score = trust * 2 + (5 - min(d_km, 50) / 10) + experience * 0.5
        ranked.append({
            "worker": w_ser,
            "distance_km": round(d_km, 2),
            "score": round(score, 2),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:20]


class ConfirmAssignmentInput(BaseModel):
    job_id: str
    worker_ids: List[str]


@app.post("/jobs/confirm")
def confirm_assignment(inp: ConfirmAssignmentInput):
    from bson import ObjectId
    try:
        job = db["farmerjob"].find_one({"_id": ObjectId(inp.job_id)})
    except Exception:
        job = db["farmerjob"].find_one({"_id": inp.job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    confirmed = []
    for wid in inp.worker_ids:
        assign = JobAssignment(job_id=inp.job_id, worker_id=wid, status="confirmed", locked_at=datetime.utcnow())
        create_document("jobassignment", assign)
        confirmed.append(wid)

    db["farmerjob"].update_one(
        {"_id": job["_id"]},
        {
            "$set": {
                "status": "confirmed",
                "confirmed_worker_ids": confirmed,
            }
        },
    )
    return {"status": "ok", "confirmed": confirmed}


# Worker profile endpoints
class RegisterWorkerInput(BaseModel):
    user_id: str
    skills: List[str]
    wage_expectation: float
    availability_dates: List[str]
    max_distance_km: float
    preferences: Optional[List[str]] = None
    home_location: Optional[GeoPoint] = None


@app.post("/workers")
def register_worker(inp: RegisterWorkerInput):
    profile = WorkerProfile(**inp.model_dump())
    # Upsert by user_id
    db["workerprofile"].update_one(
        {"user_id": profile.user_id}, {"$set": profile.model_dump()}, upsert=True
    )
    doc = db["workerprofile"].find_one({"user_id": profile.user_id})
    return {"worker": serialize_doc(doc)}


@app.get("/workers/nearby")
def workers_nearby(lat: float = Query(...), lon: float = Query(...), task: Optional[str] = None, radius_km: float = 50.0):
    workers = [serialize_doc(w) for w in db["workerprofile"].find({})]
    result = []
    for w in workers:
        home = w.get("home_location") or {}
        if not home:
            continue
        d = haversine_km(lat, lon, home.get("lat", lat), home.get("lon", lon))
        if d <= radius_km:
            if task and task.lower() not in [s.lower() for s in w.get("skills", [])]:
                continue
            w["distance_km"] = round(d, 2)
            result.append(w)
    result.sort(key=lambda x: x.get("distance_km", 9999))
    return {"workers": result[:50]}


# -----------------------
# 3. Crop Price Data & Forecasting (mocked)
# -----------------------

@app.post("/market/refresh")
def market_refresh():
    # Mock some prices for a few commodities and regions
    now = datetime.utcnow()
    data = []
    for commodity in ["ragi", "tomato", "paddy", "chilli"]:
        for region in ["KA", "TN", "MH"]:
            price = 20 + hash(commodity + region) % 30
            tick = PriceTick(
                commodity=commodity,
                region=region,
                price=float(price),
                timestamp=now,
                volatility=round((hash(region) % 10) / 10.0, 2),
                source="mock",
            )
            create_document("pricetick", tick)
            data.append(tick.model_dump())
    return {"inserted": len(data)}


@app.get("/market/forecast")
def market_forecast(commodity: str, region: str, horizon_days: int = 15):
    base_price = 30 + (hash(commodity + region) % 20)
    trend_dir = ["up", "down", "flat"][hash(commodity) % 3]
    delta = 0.5 if trend_dir == "up" else (-0.5 if trend_dir == "down" else 0.0)

    curve = []
    price = float(base_price)
    for i in range(horizon_days):
        price = max(1.0, price + delta + ((hash((commodity, region, i)) % 5) - 2) * 0.1)
        day = (datetime.utcnow() + timedelta(days=i + 1)).date().isoformat()
        curve.append({"date": day, "price": round(price, 2)})

    alerts = []
    if trend_dir == "up":
        alerts.append("Price likely to rise")
    elif trend_dir == "down":
        alerts.append("Risk of price drop")

    forecast = PriceForecast(
        commodity=commodity,
        region=region,
        horizon_days=horizon_days,
        forecast_curve=curve,
        trend=trend_dir,  # type: ignore
        risk_alerts=alerts,
    )
    return forecast


# -----------------------
# 4. Crop Recommendation Engine (rule-based demo)
# -----------------------

class AdvisoryInput(BaseModel):
    region: str
    season: Optional[str] = None
    soil_type: Optional[str] = None
    expected_rainfall_mm: Optional[float] = None


@app.post("/advisory/recommend")
def advisory_recommend(inp: AdvisoryInput):
    candidates = ["ragi", "paddy", "tomato", "chilli", "groundnut"]
    recs = []
    for crop in candidates:
        fc = market_forecast(crop, inp.region, 15)
        last = fc.forecast_curve[-1]["price"]
        first = fc.forecast_curve[0]["price"]
        trend_score = (last - first) / max(1.0, first)
        weather_bonus = 0.1 if (inp.expected_rainfall_mm or 0) > 100 and crop in {"paddy"} else 0.0
        soil_bonus = 0.1 if (inp.soil_type or "").lower() in ("red", "laterite") and crop in {"groundnut", "ragi"} else 0.0
        score = trend_score + weather_bonus + soil_bonus
        recs.append({
            "crop": crop,
            "score": round(score, 3),
            "expected_price": fc.forecast_curve[-1]["price"],
            "workers_needed": 5 if crop in {"paddy", "tomato"} else 3,
            "rationale": [
                f"Trend: {fc.trend}",
                *(fc.risk_alerts),
            ],
        })
    recs.sort(key=lambda x: x["score"], reverse=True)
    return CropRecommendation(region=inp.region, season=inp.season, recommended=recs[:3])


# -----------------------
# 5. Buyer Network & Messaging
# -----------------------

@app.post("/buyers/request")
def create_buyer_request(req: BuyerRequest):
    req_id = create_document("buyerrequest", req)
    return {"request_id": req_id}


class SendMessageInput(BaseModel):
    from_id: str
    to_id: str
    content: str


@app.post("/chat/send")
def send_message(inp: SendMessageInput):
    msg = ChatMessage(
        from_id=inp.from_id, to_id=inp.to_id, content=inp.content, encrypted=True, sent_at=datetime.utcnow()
    )
    mid = create_document("chatmessage", msg)
    return {"message_id": mid}


@app.get("/chat/with")
def chat_with(user_id: str, peer_id: str):
    msgs = list(
        db["chatmessage"].find(
            {
                "$or": [
                    {"from_id": user_id, "to_id": peer_id},
                    {"from_id": peer_id, "to_id": user_id},
                ]
            }
        ).sort("sent_at", 1)
    )
    return {"messages": [serialize_doc(m) for m in msgs]}


# -----------------------
# 6. Notifications & Sync (Low network support)
# -----------------------

@app.post("/notify")
def notify(n: Notification):
    nid = create_document("notification", n)
    return {"notification_id": nid, "sent": True}


@app.post("/sync/push")
def sync_push(event: SyncEvent):
    sid = create_document("syncevent", event)
    return {"synced": True, "id": sid}


@app.get("/sync/pull")
def sync_pull(user_id: str):
    items = [serialize_doc(x) for x in get_documents("syncevent", {"user_id": user_id}, limit=100)]
    return {"events": items}


# Convenience endpoints
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    from bson import ObjectId
    try:
        doc = db["farmerjob"].find_one({"_id": ObjectId(job_id)})
    except Exception:
        doc = db["farmerjob"].find_one({"_id": job_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": serialize_doc(doc)}


@app.get("/jobs")
def list_jobs(farmer_id: Optional[str] = None, status: Optional[str] = None):
    q: Dict[str, Any] = {}
    if farmer_id:
        q["farmer_id"] = farmer_id
    if status:
        q["status"] = status
    items = [serialize_doc(x) for x in db["farmerjob"].find(q).sort("created_at", -1)]
    return {"jobs": items}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
