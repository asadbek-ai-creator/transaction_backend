"""
Live demo support for AML Alert Escalation.

This module is ONLY for presentation/demo purposes (e.g. showing judges a
live "phone transfer -> risk score appears on dashboard" flow). It is not
part of the competition submission pipeline.

It provides:
  - an approximate so'm <-> miqdor_indeksi calibration (the real formula
    behind the competition's standardized index is unknown/undisclosed;
    this is a reasonable, clearly-labeled approximation for demo purposes)
  - a deterministic-per-card fake transaction history generator, so each
    demo "card number" behaves like a consistent persona across replays
  - a tiny in-memory store for the latest demo result + a short feed,
    polled by the dashboard's "Live Demo" tab
"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Approximate so'm <-> miqdor_indeksi calibration
# --------------------------------------------------------------------------
# Anchor points: (miqdor_indeksi quantile observed in the real dataset,
# an assumed real-world so'm amount at that quantile). The index anchors
# come from the actual train_transactions distribution; the so'm values
# are an assumption for demo purposes only -- the organizers never
# disclosed the real conversion.
_ANCHORS_INDEX = [-2.9, -2.12, -1.54, -0.80, -0.24, 0.43, 1.64, 2.64, 6.7]
_ANCHORS_SOM = [20_000, 50_000, 150_000, 500_000, 1_200_000, 3_000_000, 15_000_000, 50_000_000, 500_000_000]
_LOG_ANCHORS_SOM = [math.log(v) for v in _ANCHORS_SOM]


def som_to_index(amount_som: float) -> float:
    """Approximate so'm -> miqdor_indeksi via log-linear interpolation over
    empirical anchors. Clearly an approximation - see module docstring."""
    amount_som = max(amount_som, 1000)
    log_amt = math.log(amount_som)
    return float(np.interp(log_amt, _LOG_ANCHORS_SOM, _ANCHORS_INDEX))


def index_to_som_approx(index: float) -> float:
    """Inverse mapping, mostly for display/debugging."""
    log_amt = float(np.interp(index, _ANCHORS_INDEX, _LOG_ANCHORS_SOM))
    return math.exp(log_amt)


# --------------------------------------------------------------------------
# Fake background history per "card" (deterministic by card number so a
# demo can be repeated consistently)
# --------------------------------------------------------------------------
TX_TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
TYPE_WEIGHTS = [0.55, 0.30, 0.13, 0.02]


def _rng_for_card(card_number: str) -> np.random.Generator:
    seed = int(hashlib.sha256(card_number.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def generate_background_history(card_number: str, now: datetime, n_tx: int = 22) -> pd.DataFrame:
    """A believable 120-day transaction history for a fake demo customer,
    seeded deterministically from the card number entered on the phone UI."""
    rng = _rng_for_card(card_number)

    days_ago = rng.uniform(0.5, 120, size=n_tx)
    timestamps = [now - timedelta(days=float(d), hours=float(rng.uniform(0, 24))) for d in days_ago]

    directions = rng.choice(["kirim", "chiqim"], size=n_tx, p=[0.7, 0.3])
    types = rng.choice(TX_TYPES, size=n_tx, p=TYPE_WEIGHTS)
    # everyday amounts: mostly near the dataset's typical range
    amounts = rng.normal(loc=-0.15, scale=0.6, size=n_tx)
    amounts = np.clip(amounts, -2.8, 2.0)

    df = pd.DataFrame({
        "tranzaksiya_vaqti": timestamps,
        "kirim_chiqim": directions,
        "tranzaksiya_turi": types,
        "miqdor_indeksi": amounts,
    })
    return df.sort_values("tranzaksiya_vaqti").reset_index(drop=True)


def build_live_burst(now: datetime, amount_som: float, tx_type: str) -> pd.DataFrame:
    """The visible, presenter-triggered transfer plus two smaller synthetic
    transfers earlier the same day, of the same risky type. This mirrors a
    classic AML red flag (several transfers in a short window, sometimes
    used to split up a larger amount / "structuring") and gives the model
    a much more visible, realistic signal than one isolated transaction —
    matching what the EDA found: recent bursts of activity, not single
    transactions, are what correlates with escalation."""
    base_index = som_to_index(amount_som)
    rows = [
        {
            "tranzaksiya_vaqti": now - timedelta(hours=3, minutes=10),
            "kirim_chiqim": "chiqim",
            "tranzaksiya_turi": tx_type,
            "miqdor_indeksi": base_index * 0.55,
        },
        {
            "tranzaksiya_vaqti": now - timedelta(hours=1, minutes=5),
            "kirim_chiqim": "chiqim",
            "tranzaksiya_turi": tx_type,
            "miqdor_indeksi": base_index * 0.7,
        },
        {
            "tranzaksiya_vaqti": now,
            "kirim_chiqim": "chiqim",
            "tranzaksiya_turi": tx_type,
            "miqdor_indeksi": base_index,
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# In-memory demo state (single-process; fine for a live local demo)
# --------------------------------------------------------------------------
class DemoEntry(Dict):
    pass


_FEED: Deque[dict] = deque(maxlen=20)


def push_demo_result(entry: dict) -> None:
    _FEED.appendleft(entry)


def get_latest() -> Optional[dict]:
    return _FEED[0] if _FEED else None


def get_feed(limit: int = 10) -> List[dict]:
    return list(_FEED)[:limit]


def mask_card(card_number: str) -> str:
    digits = "".join(ch for ch in card_number if ch.isdigit())
    if len(digits) < 4:
        return "**** ****"
    return f"**** {digits[-4:]}"
