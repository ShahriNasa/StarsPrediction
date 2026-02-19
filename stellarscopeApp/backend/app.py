import base64
import os
from datetime import datetime
from io import BytesIO
from pdb import run
from typing import List, Dict, Any, Optional

import numpy as np
import joblib
import requests
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlmodel import SQLModel, Field, Session, create_engine, select


# -------------------------
# Device
# -------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_built()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

CLOUD_URL = "https://europe-west3-astronomylight.cloudfunctions.net/LightCurve"


# -------------------------
# SQLite (SQLModel)
# -------------------------
DB_PATH = os.getenv("STELLARSCOPE_DB", "./data/stellarscope.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    note: str = Field(default="")


class StarFile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)

    filename: str
    star_display: str = Field(default="", index=True)

    # helps dedupe if user submits same file again
    content_sha256: str = Field(default="", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    star_file_id: int = Field(foreign_key="starfile.id", index=True)

    day: str = Field(index=True)
    property: str = Field(index=True)
    value: float

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class PredictionError(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    star_file_id: int = Field(foreign_key="starfile.id", index=True)

    day: str = Field(index=True)
    property: str = Field(index=True)
    error: str

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


# -------------------------
# App
# -------------------------
app = FastAPI(title="StellarScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Request schema (JSON /predict)
# -------------------------
class PredictRequest(BaseModel):
    properties: List[str]
    days: List[str]


# -------------------------
# Request schema (Form /predict2)
# -------------------------
class BulkDeleteRunsRequest(BaseModel):
    run_ids: List[int]

# -------------------------
# Request schema (Form /delete_stars)
# -------------------------
class StarDeleteItem(BaseModel):
    run_id: int
    star_file_id: int

# -------------------------
# Request schema (Form /delete_stars_bulk)
# -------------------------
class BulkDeleteStarsRequest(BaseModel):
    stars: List[StarDeleteItem]


# -------------------------
# Helpers (model/scaler decode)
# -------------------------
def load_model_from_base64(encoded_model_data: str, device: torch.device):
    model_bytes = base64.b64decode(encoded_model_data)
    model_buffer = BytesIO(model_bytes)
    # SECURITY NOTE: torch.load unpickles -> only safe if upstream is trusted.
    model = torch.load(model_buffer, map_location=device)
    model.to(device)
    model.eval()
    return model


def load_scaler_from_base64(encoded_scaler_data: str):
    scaler_bytes = base64.b64decode(encoded_scaler_data)
    scaler_buffer = BytesIO(scaler_bytes)
    scaler = joblib.load(scaler_buffer)
    return scaler


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")


def star_display_name(filename: str) -> str:
    # filename like: kplr007700670-2010355172524_llc.fits
    base = filename
    if base.lower().endswith(".fits"):
        base = base[:-5]
    import re
    m = re.match(r"^(kplr)(\d+)-(\d+)_([a-z]+)$", base, flags=re.I)
    if m:
        mission, kid_raw, q_raw, cad_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        kid = str(int(kid_raw))
        mission_name = "Kepler" if mission.lower() == "kplr" else mission.upper()
        cadence = "LLC" if cad_raw.lower() == "llc" else cad_raw.upper()
        return f"{mission_name} {kid} (Q{q_raw} • {cadence})"
    if len(base) > 28:
        return base[:20] + "…" + base[-7:]
    return base


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _delete_starfile_cascade(session: Session, star_file_id: int) -> None:
    # delete children first
    preds = session.exec(select(Prediction).where(Prediction.star_file_id == star_file_id)).all()
    for p in preds:
        session.delete(p)

    errs = session.exec(select(PredictionError).where(PredictionError.star_file_id == star_file_id)).all()
    for e in errs:
        session.delete(e)

    star = session.get(StarFile, star_file_id)
    if star:
        session.delete(star)


def _delete_run_cascade(session: Session, run_id: int) -> int:
    run = session.get(Run, run_id)
    if not run:
        return 0

    stars = session.exec(select(StarFile).where(StarFile.run_id == run_id)).all()
    for s in stars:
        _delete_starfile_cascade(session, int(s.id))

    session.delete(run)
    return 1

# -------------------------
# API
# -------------------------
@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True, "device": str(device), "db": DB_PATH}


# ---- Run history endpoints ----
@app.post("/runs")
def create_run(note: str = Form(default="")):
    with Session(engine) as session:
        r = Run(note=note or "")
        session.add(r)
        session.commit()
        session.refresh(r)
        return {"run_id": r.id, "created_at": r.created_at.isoformat(), "note": r.note}


@app.get("/runs")
def list_runs(limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    with Session(engine) as session:
        runs = session.exec(
            select(Run).order_by(Run.created_at.desc()).offset(offset).limit(limit)
        ).all()
        return [
            {"run_id": r.id, "created_at": r.created_at.isoformat(), "note": r.note}
            for r in runs
        ]

@app.post("/runs/bulk-delete")
def bulk_delete_runs(req: BulkDeleteRunsRequest):
    if not req.run_ids:
        raise HTTPException(status_code=400, detail="run_ids is empty")

    deleted = 0
    with Session(engine) as session:
        try:
            for rid in req.run_ids:
                deleted += _delete_run_cascade(session, int(rid))
            session.commit()
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    return {"deleted_runs": deleted}

@app.post("/stars/bulk-delete")
def bulk_delete_stars(req: BulkDeleteStarsRequest):
    if not req.stars:
        raise HTTPException(status_code=400, detail="stars is empty")

    deleted = 0
    with Session(engine) as session:
        try:
            for item in req.stars:
                # safety: ensure star belongs to the given run_id
                star = session.get(StarFile, int(item.star_file_id))
                if not star:
                    continue
                if int(star.run_id) != int(item.run_id):
                    continue

                _delete_starfile_cascade(session, int(star.id))
                deleted += 1

            session.commit()
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    return {"deleted_stars": deleted}
@app.get("/runs/{run_id}")
def get_run(run_id: int):
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")

        stars = session.exec(
            select(StarFile).where(StarFile.run_id == run_id).order_by(StarFile.created_at.desc())
        ).all()

        # Build star -> predictions/errors
        out_stars = []
        for s in stars:
            preds = session.exec(
                select(Prediction).where(Prediction.star_file_id == s.id).order_by(Prediction.day, Prediction.property)
            ).all()
            errs = session.exec(
                select(PredictionError).where(PredictionError.star_file_id == s.id).order_by(PredictionError.day, PredictionError.property)
            ).all()

            out_stars.append({
                "star_file_id": s.id,
                "filename": s.filename,
                "star_display": s.star_display,
                "content_sha256": s.content_sha256,
                "created_at": s.created_at.isoformat(),
                "results": [{"day": p.day, "property": p.property, "value": p.value} for p in preds],
                "errors": [{"day": e.day, "property": e.property, "error": e.error} for e in errs],
            })

        return {
            "run_id": run.id,
            "created_at": run.created_at.isoformat(),
            "note": run.note,
            "stars": out_stars,
        }

@app.delete("/runs/{run_id}")
def delete_run(run_id: int):
    with Session(engine) as session:
        n = _delete_run_cascade(session, run_id)
        if n == 0:
            raise HTTPException(404, "Run not found")
        session.commit()
    return {"deleted": True, "run_id": run_id}


@app.delete("/stars/{star_file_id}")
def delete_star(star_file_id: int):
    with Session(engine) as session:
        star = session.get(StarFile, star_file_id)
        if not star:
            raise HTTPException(404, "Star file not found")
        _delete_starfile_cascade(session, star_file_id)
        session.commit()
    return {"deleted": True, "star_file_id": star_file_id}
# ---- Core prediction logic (shared) ----
async def _predict_logic(meta: PredictRequest, fits_file: UploadFile) -> Dict[str, Any]:
    if not meta.properties:
        raise HTTPException(status_code=400, detail="Select at least one property.")
    if not meta.days:
        raise HTTPException(status_code=400, detail="Select at least one day.")
    if not fits_file.filename:
        raise HTTPException(status_code=400, detail="No FITS file uploaded.")

    raw = await fits_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    encoded_fits_file = base64.b64encode(raw).decode("utf-8")

    results = []
    errors = []

    for day in meta.days:
        for prop in meta.properties:
            payload = {
                "fits_file": encoded_fits_file,
                "properties": b64(prop),
                "days": b64(str(day)),
            }

            try:
                r = requests.post(CLOUD_URL, json=payload, timeout=120)
            except requests.RequestException as e:
                errors.append({"day": str(day), "property": prop, "error": f"Request failed: {e}"})
                continue

            if r.status_code != 200:
                errors.append({"day": str(day), "property": prop, "error": f"Upstream status {r.status_code}"})
                continue

            try:
                data = r.json()
                x_values = np.array(data["x_values"])
                std_values = np.array(data["std_values"])
                model = load_model_from_base64(data["model_file"], device)
                scaler = load_scaler_from_base64(data["scaler_file"])
            except Exception as e:
                errors.append({"day": str(day), "property": prop, "error": f"Bad upstream payload: {e}"})
                continue

            try:
                x_values_tensor = torch.tensor(x_values.T, dtype=torch.float32, device=device).unsqueeze(1)
                std_values_tensor = torch.tensor(std_values, dtype=torch.float32, device=device)

                with torch.no_grad():
                    preds = model(x_values_tensor, std_values_tensor).detach().cpu().numpy().flatten()

                final_preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                value = float(final_preds[0])

                results.append({"day": str(day), "property": prop, "value": value})
            except Exception as e:
                errors.append({"day": str(day), "property": prop, "error": f"Inference failed: {e}"})

    return {
        "filename": fits_file.filename,
        "star_display": star_display_name(fits_file.filename),
        "sha256": sha256_bytes(raw),
        "results": results,
        "errors": errors,
    }


@app.post("/predict")
async def predict(meta: PredictRequest, fits_file: UploadFile = File(...)) -> Dict[str, Any]:
    return await _predict_logic(meta, fits_file)


@app.post("/predict2")
async def predict2(
    fits_file: UploadFile = File(...),
    properties: List[str] = Form(...),
    days: List[str] = Form(...),
    run_id: int = Form(...),
    note: str = Form(default=""),
):
    """
    If run_id is provided -> store under that run.
    Else -> create a new run for this request (one file = one run).
    """
    req = PredictRequest(properties=properties, days=days)
    payload = await _predict_logic(req, fits_file)

    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(400, detail="run_id does not exist")

        existing = session.exec(
            select(StarFile).where(
                StarFile.run_id == run_id,
                StarFile.content_sha256 == payload["sha256"],
            )
        ).first()

        if existing:
            star = existing
        else:
            star = StarFile(
                run_id=run_id,
                filename=payload["filename"],
                star_display=payload["star_display"],
                content_sha256=payload["sha256"],
            )
            session.add(star)
            session.commit()
            session.refresh(star)

        # capture IDs while attached (critical)
        star_id = int(star.id)

        # delete old
        for p in session.exec(select(Prediction).where(Prediction.star_file_id == star_id)).all():
            session.delete(p)
        for e in session.exec(select(PredictionError).where(PredictionError.star_file_id == star_id)).all():
            session.delete(e)
        session.commit()

        # insert new
        for r in payload["results"]:
            session.add(Prediction(
                star_file_id=star_id,
                day=str(r["day"]),
                property=str(r["property"]),
                value=float(r["value"]),
            ))
        for e in payload["errors"]:
            session.add(PredictionError(
                star_file_id=star_id,
                day=str(e["day"]),
                property=str(e["property"]),
                error=str(e["error"]),
            ))
        session.commit()

    return {
        "run_id": run_id,
        "star_file_id": star_id,
        **payload,
    }
