"""
Feature engineering for AML alert escalation scoring.

Turns a signal's raw transaction history into the ~79-column feature
vector the LightGBM model was trained on. Mirrors the pipeline used
during training exactly (see the research notebook) so that scores
from this module match the model's training-time behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy

TX_TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
RECENCY_WINDOWS = [1, 3, 7, 14, 30, 60, 90, 120]


def build_features(tx: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """
    tx: columns = signal_id, tranzaksiya_vaqti (datetime), kirim_chiqim,
        tranzaksiya_turi, miqdor_indeksi
    sig: columns = signal_id, signal_sanasi (datetime)

    Returns one row per signal_id with all engineered features.
    """
    m = tx.merge(sig[["signal_id", "signal_sanasi"]], on="signal_id")
    m["days_before_signal"] = (
        m["signal_sanasi"] - m["tranzaksiya_vaqti"]
    ).dt.total_seconds() / 86400
    m["hour"] = m["tranzaksiya_vaqti"].dt.hour
    m["dow"] = m["tranzaksiya_vaqti"].dt.dayofweek
    m["is_night"] = m["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    m["is_weekend"] = (m["dow"] >= 5).astype(int)
    m["is_kirim"] = (m["kirim_chiqim"] == "kirim").astype(int)

    g = m.groupby("signal_id")
    feats = pd.DataFrame(index=g.size().index)
    feats.index.name = "signal_id"

    # --- Volume / distribution stats ---
    feats["n_tx"] = g.size()
    feats["mean_amt"] = g["miqdor_indeksi"].mean()
    feats["sum_amt"] = g["miqdor_indeksi"].sum()
    feats["std_amt"] = g["miqdor_indeksi"].std()
    feats["max_amt"] = g["miqdor_indeksi"].max()
    feats["min_amt"] = g["miqdor_indeksi"].min()
    feats["median_amt"] = g["miqdor_indeksi"].median()
    feats["q10_amt"] = g["miqdor_indeksi"].quantile(0.1)
    feats["q90_amt"] = g["miqdor_indeksi"].quantile(0.9)
    feats["range_amt"] = feats["max_amt"] - feats["min_amt"]
    feats["iqr_amt"] = g["miqdor_indeksi"].quantile(0.75) - g["miqdor_indeksi"].quantile(0.25)
    feats["skew_amt"] = g["miqdor_indeksi"].apply(lambda x: skew(x) if len(x) > 2 else 0)
    feats["kurt_amt"] = g["miqdor_indeksi"].apply(lambda x: kurtosis(x) if len(x) > 3 else 0)
    feats["cv_amt"] = feats["std_amt"] / (feats["mean_amt"].abs() + 1e-6)

    # --- Flow (kirim/chiqim) ---
    feats["kirim_share"] = g["is_kirim"].mean()
    kirim_sum = m[m["is_kirim"] == 1].groupby("signal_id")["miqdor_indeksi"].sum()
    chiqim_sum = m[m["is_kirim"] == 0].groupby("signal_id")["miqdor_indeksi"].sum()
    feats["kirim_sum"] = kirim_sum
    feats["chiqim_sum"] = chiqim_sum
    feats[["kirim_sum", "chiqim_sum"]] = feats[["kirim_sum", "chiqim_sum"]].fillna(0)
    feats["net_flow"] = feats["kirim_sum"] - feats["chiqim_sum"]
    feats["flow_ratio"] = feats["kirim_sum"] / (feats["chiqim_sum"].abs() + 1e-6)

    kirim_cnt = m[m["is_kirim"] == 1].groupby("signal_id").size()
    chiqim_cnt = m[m["is_kirim"] == 0].groupby("signal_id").size()
    feats["kirim_cnt"] = kirim_cnt
    feats["chiqim_cnt"] = chiqim_cnt
    feats[["kirim_cnt", "chiqim_cnt"]] = feats[["kirim_cnt", "chiqim_cnt"]].fillna(0)
    feats["kirim_chiqim_cnt_ratio"] = feats["kirim_cnt"] / (feats["chiqim_cnt"] + 1e-6)

    # --- Transaction type ---
    for t in TX_TYPES:
        cnt = m[m["tranzaksiya_turi"] == t].groupby("signal_id").size()
        feats[f"{t}_share"] = (cnt / feats["n_tx"]).fillna(0)
        amt = m[m["tranzaksiya_turi"] == t].groupby("signal_id")["miqdor_indeksi"].sum()
        feats[f"{t}_sum"] = amt.reindex(feats.index).fillna(0)
        mean_amt_t = m[m["tranzaksiya_turi"] == t].groupby("signal_id")["miqdor_indeksi"].mean()
        feats[f"{t}_mean"] = mean_amt_t.reindex(feats.index).fillna(0)

    def type_entropy(x):
        p = x["tranzaksiya_turi"].value_counts(normalize=True).values
        return entropy(p)

    feats["type_entropy"] = g.apply(type_entropy)

    # --- Recency windows ---
    for window in RECENCY_WINDOWS:
        sub = m[m["days_before_signal"] <= window]
        gg = sub.groupby("signal_id").size()
        feats[f"n_tx_last{window}d"] = gg.reindex(feats.index).fillna(0)
        amt = sub.groupby("signal_id")["miqdor_indeksi"].sum()
        feats[f"sum_amt_last{window}d"] = amt.reindex(feats.index).fillna(0)
        feats[f"rate_last{window}d"] = feats[f"n_tx_last{window}d"] / window

    feats["recent_ratio_7_30"] = feats["n_tx_last7d"] / (feats["n_tx_last30d"] + 1e-6)
    feats["recent_ratio_1_7"] = feats["n_tx_last1d"] / (feats["n_tx_last7d"] + 1e-6)
    feats["recent_ratio_30_90"] = feats["n_tx_last30d"] / (feats["n_tx_last90d"] + 1e-6)
    feats["accel_7_vs_rest"] = feats["rate_last7d"] / (feats["rate_last90d"] + 1e-6)
    feats["tx_density"] = feats["n_tx"] / 180.0

    # --- Timing patterns ---
    feats["night_share"] = g["is_night"].mean()
    feats["weekend_share"] = g["is_weekend"].mean()

    def gap_std(x):
        t = x["tranzaksiya_vaqti"].sort_values()
        if len(t) < 3:
            return 0.0
        gaps = t.diff().dt.total_seconds().dropna() / 3600.0
        return gaps.std()

    def gap_mean(x):
        t = x["tranzaksiya_vaqti"].sort_values()
        if len(t) < 3:
            return 0.0
        gaps = t.diff().dt.total_seconds().dropna() / 3600.0
        return gaps.mean()

    def gap_min(x):
        t = x["tranzaksiya_vaqti"].sort_values()
        if len(t) < 3:
            return 0.0
        gaps = t.diff().dt.total_seconds().dropna() / 3600.0
        return gaps.min()

    feats["gap_std"] = g.apply(gap_std)
    feats["gap_mean"] = g.apply(gap_mean)
    feats["gap_min"] = g.apply(gap_min)

    span = g["tranzaksiya_vaqti"].agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)
    feats["activity_span_days"] = span
    feats["tx_per_active_day"] = feats["n_tx"] / (feats["activity_span_days"] + 1)
    feats["unique_days_active"] = g["tranzaksiya_vaqti"].apply(lambda x: x.dt.date.nunique())
    feats["days_active_ratio"] = feats["unique_days_active"] / (feats["activity_span_days"] + 1)

    feats["days_since_last_tx"] = g["days_before_signal"].min()
    feats["days_since_first_tx"] = g["days_before_signal"].max()

    def daily_trend(x):
        daily = x.groupby(x["tranzaksiya_vaqti"].dt.date).size()
        if len(daily) < 3:
            return 0.0
        idx = np.arange(len(daily))
        return np.polyfit(idx, daily.values, 1)[0]

    feats["daily_count_trend"] = g.apply(daily_trend)

    # --- Signal date metadata ---
    sig2 = sig.copy()
    sig2["sig_dow"] = sig2["signal_sanasi"].dt.dayofweek
    sig2["sig_month"] = sig2["signal_sanasi"].dt.month
    sig2["sig_day"] = sig2["signal_sanasi"].dt.day

    feats = feats.reset_index()
    feats = feats.merge(sig2[["signal_id", "sig_dow", "sig_month", "sig_day"]], on="signal_id", how="left")
    return feats
