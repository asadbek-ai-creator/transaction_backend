"""
AML Alert Escalation Risk API
Team IT-Ledi

Serves the trained LightGBM model behind two endpoints:
  POST /predict        - score a single alert from its transaction list
  POST /predict/batch   - score a whole test set from two CSV files

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Docs available at http://localhost:8000/docs
"""
from __future__ import annotations

import io
import pickle
from datetime import datetime
from enum import Enum
from typing import List, Optional

# lightgbm must be imported before pandas/numpy: on Windows its bundled OpenMP
# runtime conflicts with the one numpy/scipy loads, and whichever comes second
# ends up with a broken thread pool, making LGBM_BoosterPredictForMat segfault
# ("access violation reading 0x0"). The model arrives via pickle.load below,
# which would otherwise import lightgbm too late.
import lightgbm  # noqa: F401  (import for side effect - see comment above)
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, conint, confloat

import demo
from features import build_features

# --------------------------------------------------------------------------
# App & model loading
# --------------------------------------------------------------------------

app = FastAPI(
    title="AML Alert Escalation Risk API",
    description="Team IT-Ledi — scores financial-monitoring alerts by escalation probability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model_full_data.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS: List[str] = pickle.load(f)

MIN_RECOMMENDED_TX = 5  # below this, predictions are shown with a low-confidence flag


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class Direction(str, Enum):
    kirim = "kirim"
    chiqim = "chiqim"


class TxType(str, Enum):
    karta = "karta"
    bank_otkazmasi = "bank_otkazmasi"
    naqd = "naqd"
    xalqaro = "xalqaro"


class Transaction(BaseModel):
    tranzaksiya_vaqti: datetime = Field(..., description="Transaction timestamp, ISO-8601")
    kirim_chiqim: Direction
    tranzaksiya_turi: TxType
    miqdor_indeksi: float = Field(..., description="Standardized transaction size index")


class PredictRequest(BaseModel):
    signal_id: str = Field(default="SG_MANUAL", description="Optional label for this alert")
    signal_sanasi: datetime = Field(..., description="Alert date, ISO-8601")
    transactions: List[Transaction] = Field(..., min_items=1)


class DemoTransferRequest(BaseModel):
    card_number: str = Field(..., description="Card number typed on the phone UI (demo only)")
    amount_som: float = Field(..., gt=0, description="Transfer amount in so'm")
    tranzaksiya_turi: TxType = Field(default=TxType.karta)


class TopFeature(BaseModel):
    name: str
    value: float


class PredictResponse(BaseModel):
    signal_id: str
    ehtimollik: confloat(ge=0, le=1)
    risk_level: str
    n_transactions: int
    low_confidence: bool
    top_features: List[TopFeature]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def risk_level(p: float) -> str:
    if p < 0.15:
        return "low"
    if p < 0.35:
        return "medium"
    return "high"


def score_dataframe(tx_df: pd.DataFrame, sig_df: pd.DataFrame) -> pd.DataFrame:
    """tx_df, sig_df -> dataframe with signal_id, ehtimollik, n_tx"""
    feats = build_features(tx_df, sig_df)
    # guarantee column order / presence matches training
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0
    X = feats[FEATURE_COLUMNS]
    preds = MODEL.predict(X)
    out = pd.DataFrame({
        "signal_id": feats["signal_id"],
        "ehtimollik": preds,
        "n_tx": feats["n_tx"],
    })
    return out


def top_contributing_features(feats_row: pd.Series, n: int = 5) -> List[TopFeature]:
    """Rank this alert's features by global model importance for a quick explanation panel."""
    importances = MODEL.feature_importance(importance_type="gain")
    imp_series = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
    top_names = imp_series.head(n).index.tolist()
    return [TopFeature(name=name, value=float(feats_row[name])) for name in top_names]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(FEATURE_COLUMNS)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.transactions) == 0:
        raise HTTPException(400, "At least one transaction is required.")

    tx_df = pd.DataFrame([t.dict() for t in req.transactions])
    tx_df["signal_id"] = req.signal_id
    tx_df["tranzaksiya_vaqti"] = pd.to_datetime(tx_df["tranzaksiya_vaqti"])

    sig_df = pd.DataFrame({
        "signal_id": [req.signal_id],
        "signal_sanasi": [pd.to_datetime(req.signal_sanasi)],
    })

    feats = build_features(tx_df, sig_df)
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0
    X = feats[FEATURE_COLUMNS]
    proba = float(MODEL.predict(X)[0])
    n_tx = int(feats["n_tx"].iloc[0])

    return PredictResponse(
        signal_id=req.signal_id,
        ehtimollik=proba,
        risk_level=risk_level(proba),
        n_transactions=n_tx,
        low_confidence=n_tx < MIN_RECOMMENDED_TX,
        top_features=top_contributing_features(feats.iloc[0], n=5),
    )


@app.post("/predict/batch")
async def predict_batch(
    signals_file: UploadFile = File(..., description="CSV with signal_id, signal_sanasi"),
    transactions_file: UploadFile = File(..., description="CSV or parquet with signal_id, tranzaksiya_vaqti, kirim_chiqim, tranzaksiya_turi, miqdor_indeksi"),
):
    sig_bytes = await signals_file.read()
    tx_bytes = await transactions_file.read()

    try:
        sig_df = pd.read_csv(io.BytesIO(sig_bytes), parse_dates=["signal_sanasi"])
    except Exception as e:
        raise HTTPException(400, f"Could not parse signals file: {e}")

    try:
        if transactions_file.filename.endswith(".parquet"):
            tx_df = pd.read_parquet(io.BytesIO(tx_bytes))
        else:
            tx_df = pd.read_csv(io.BytesIO(tx_bytes), parse_dates=["tranzaksiya_vaqti"])
    except Exception as e:
        raise HTTPException(400, f"Could not parse transactions file: {e}")

    required_sig_cols = {"signal_id", "signal_sanasi"}
    required_tx_cols = {"signal_id", "tranzaksiya_vaqti", "kirim_chiqim", "tranzaksiya_turi", "miqdor_indeksi"}
    if not required_sig_cols.issubset(sig_df.columns):
        raise HTTPException(400, f"signals file missing columns: {required_sig_cols - set(sig_df.columns)}")
    if not required_tx_cols.issubset(tx_df.columns):
        raise HTTPException(400, f"transactions file missing columns: {required_tx_cols - set(tx_df.columns)}")

    result = score_dataframe(tx_df, sig_df)

    submission = result[["signal_id", "ehtimollik"]].copy()

    # sanity checks mirroring the competition rules
    if submission["signal_id"].duplicated().any():
        raise HTTPException(500, "Duplicate signal_id produced internally — check input data.")
    if not submission["ehtimollik"].between(0, 1).all():
        raise HTTPException(500, "Predictions outside [0,1] — unexpected model output.")

    buf = io.StringIO()
    submission.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.post("/predict/batch/preview")
async def predict_batch_preview(
    signals_file: UploadFile = File(...),
    transactions_file: UploadFile = File(...),
    limit: conint(ge=1, le=500) = Form(50),
):
    """Same as /predict/batch but returns JSON (first `limit` rows) for on-screen review
    instead of triggering a file download — used by the dashboard's results table."""
    sig_bytes = await signals_file.read()
    tx_bytes = await transactions_file.read()

    sig_df = pd.read_csv(io.BytesIO(sig_bytes), parse_dates=["signal_sanasi"])
    if transactions_file.filename.endswith(".parquet"):
        tx_df = pd.read_parquet(io.BytesIO(tx_bytes))
    else:
        tx_df = pd.read_csv(io.BytesIO(tx_bytes), parse_dates=["tranzaksiya_vaqti"])

    result = score_dataframe(tx_df, sig_df)
    result = result.sort_values("ehtimollik", ascending=False)

    return {
        "n_signals": len(result),
        "mean_probability": float(result["ehtimollik"].mean()),
        "n_high_risk": int((result["ehtimollik"] >= 0.35).sum()),
        "rows": result.head(limit).to_dict(orient="records"),
    }


@app.get("/feature-importance")
def feature_importance(top_n: int = 15):
    importances = MODEL.feature_importance(importance_type="gain")
    imp_series = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
    top = imp_series.head(top_n)
    return [{"feature": k, "importance": float(v)} for k, v in top.items()]


# --------------------------------------------------------------------------
# Live demo endpoints (presentation only — not part of the competition
# scoring pipeline). See demo.py for the so'm<->index approximation and
# the fake background-history generator.
# --------------------------------------------------------------------------

@app.post("/demo/transfer")
def demo_transfer(req: DemoTransferRequest):
    """Called by the phone UI when the presenter taps 'Отправить'.
    Scores the (fake background history + this one live transfer) through
    the real model, stores the result for the dashboard to pick up, and
    returns only a bare confirmation — the phone screen never shows a score."""
    now = datetime.utcnow()

    history = demo.generate_background_history(req.card_number, now)
    live_tx = demo.build_live_burst(now, req.amount_som, req.tranzaksiya_turi.value)
    tx_df = pd.concat([history, live_tx], ignore_index=True)
    tx_df["signal_id"] = "DEMO"

    sig_df = pd.DataFrame({"signal_id": ["DEMO"], "signal_sanasi": [now]})

    feats = build_features(tx_df, sig_df)
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0
    proba = float(MODEL.predict(feats[FEATURE_COLUMNS])[0])

    entry = {
        "timestamp": now.isoformat(),
        "masked_card": demo.mask_card(req.card_number),
        "amount_som": req.amount_som,
        "tranzaksiya_turi": req.tranzaksiya_turi.value,
        "ehtimollik": proba,
        "risk_level": risk_level(proba),
        "n_transactions": int(feats["n_tx"].iloc[0]),
        "top_features": [f.dict() for f in top_contributing_features(feats.iloc[0], n=5)],
    }
    demo.push_demo_result(entry)

    return {"status": "ok"}


@app.get("/demo/latest")
def demo_latest():
    entry = demo.get_latest()
    if entry is None:
        return {"empty": True}
    return {"empty": False, **entry}


@app.get("/demo/feed")
def demo_feed(limit: int = 10):
    return demo.get_feed(limit)


@app.post("/demo/reset")
def demo_reset():
    """Clears the demo feed so the dashboard starts from a blank slate —
    handy between runs when presenting."""
    demo.reset_feed()
    return {"status": "reset"}
