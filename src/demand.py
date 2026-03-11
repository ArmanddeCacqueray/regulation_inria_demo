#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tensorly.decomposition import tucker
import tensorly as tl

# =============================================================
# 1. OUTILS DE CALCUL (Inchangés)
# =============================================================

def build_events(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.copy()
    ts["delta_total_bikes"] = (ts["stock"] + ts["indispo"]).diff()
    ts["delta_available_bikes"] = ts["stock"].diff()

    ts["arrivals"] = ts["delta_total_bikes"].clip(lower=0)
    ts["departures"] = (-ts["delta_total_bikes"]).clip(lower=0)

    not_regulated = ts["not_regulated"]
    not_maintenance = (ts["indispo"].diff() + 0.1 >= 0).astype(float)

    ts["true_departures"] = ts["departures"] * not_maintenance * not_regulated
    ts["true_arrivals"] = ts["arrivals"] * not_regulated

    ts["available"] = ts["delta_available_bikes"].clip(lower=0)
    ts["unavailable"] = (-ts["delta_available_bikes"]).clip(lower=0)

    ts["dock_signal"] = None
    ts.loc[ts["departures"] > 0, "dock_signal"] = 1
    ts.loc[ts["arrivals"]   > 0, "dock_signal"] = -1

    ts["bike_signal"] = None
    ts.loc[ts["available"]   > 0, "bike_signal"] = 1
    ts.loc[ts["unavailable"] > 0, "bike_signal"] = -1

    recent_dock = ts["dock_signal"].ffill().infer_objects(copy=False).shift(1).fillna(0).astype(float) > 0
    ts["fresh_dock"] = recent_dock | (ts["arrivals"] > 0)

    recent_bike = ts["bike_signal"].ffill().infer_objects(copy=False).shift(1).fillna(0).astype(float) > 0 
    ts["fresh_bike"] = recent_bike | (ts["true_departures"] > 0)
    return ts

def apply_censoring(ts: pd.DataFrame, min_stock=7, min_diapasons=5) -> pd.DataFrame:
    ts = ts.copy()
    ts["censure_vide"] = (ts["stock"] < min_stock) & (~ts["fresh_bike"])
    ts["censure_pleine"] = (ts["diapasons"] < min_diapasons) & (~ts["fresh_dock"])

    ts["demande_obs_dep"] = ts["true_departures"]
    ts.loc[ts["censure_vide"], "demande_obs_dep"] = np.nan

    ts["demande_obs_arr"] = ts["true_arrivals"]
    ts.loc[ts["censure_pleine"], "demande_obs_arr"] = np.nan
    return ts

def resample_minute(ts: pd.DataFrame, freq="1min") -> pd.DataFrame:
    return ts.set_index("time").resample(freq).ffill().reset_index()

def gaussian_nan_filter(x, sigma=5):
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x_filled = np.nan_to_num(x, nan=0.0)
    num = gaussian_filter1d(x_filled, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(mask.astype(float), sigma=sigma, mode="nearest")
    out = num / (den + 1e-8)
    out[den <= 0.1] = 0
    out[mask] = x[mask]
    return out

def add_lag_dimension(X, max_lag=6):
    S, D, T = X.shape
    XL = np.zeros((S, D, T, max_lag))
    X_flat = X.reshape(S, D * T)
    for l in range(max_lag):
        if l == 0:
            XL[..., l] = X
        else:
            lagged_flat = np.zeros_like(X_flat)
            lagged_flat[:, l:] = X_flat[:, :-l]
            XL[..., l] = lagged_flat.reshape(S, D, T)
    return XL

def tucker_reconstruction(XL, ranks):
    core, factors = tucker(XL, rank=ranks)
    return tl.tucker_to_tensor((core, factors))

def tensor_to_dataframe(X, stations, tensor_times):
    return pd.DataFrame({
        "station": np.repeat(stations, X.shape[1]),
        "time": np.tile(tensor_times, X.shape[0]),
        "demande_latente": X.reshape(-1)
    })

# =============================================================
# 2. PIPELINE PRINCIPAL (MODULARISÉ)
# =============================================================

def run_reconstruction(config):
    """
    Exécute la reconstruction Tucker.
    Prend en entrée le dictionnaire 'config' issu du JSON.
    """
    # 1. Chemins et Params depuis la config
    in_dir = Path(config["paths"]["process_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    clean_csv = in_dir / "CLEAN_last_week.csv"
    clean_20_csv = in_dir / "CLEAN_last_week_20min.csv"
    output_csv = out_dir / "RECONSTRUCTION_FINAL.csv"

    g_freq = config["params"].get("gaussian_freq", "2min")
    g_sigma = config["params"]["gaussian_sigma"]
    g_sigma = pd.to_timedelta(g_sigma).total_seconds() / pd.to_timedelta(g_freq).total_seconds()
    t_freq = config["params"].get("sample_freq", "20min")
    t_ranks = tuple(config["params"]["tucker_ranks"])
    max_lag = config["params"].get("max_lag", 5)

    print(f"--- Début Reconstruction (Sigma={g_sigma}, Ranks={t_ranks}) ---")

    # --- Load ---
    if not clean_csv.exists():
        raise FileNotFoundError(f"Il faut lancer le processing avant : {clean_csv} introuvable.")
        
    df_clean = pd.read_csv(clean_csv)
    df_clean_20 = pd.read_csv(clean_20_csv)
    df_clean["time"] = pd.to_datetime(df_clean["time"])
    df_clean_20["time"] = pd.to_datetime(df_clean_20["time"])

    # --- Mapping stations ---
    stations = np.sort(df_clean["station"].unique())
    s_to_i = {s: i for i, s in enumerate(stations)}

    # --- Build time grid ---
    s0 = stations[0]
    ts0 = df_clean[df_clean["station"] == s0].sort_values("time")
    ts0_resampled = resample_minute(ts0, g_freq)
    times = ts0_resampled["time"].values

    ts20 = resample_minute(ts0, t_freq)
    tensor_times = ts20["time"].values

    days = times.astype("datetime64[D]")
    unique_days = np.unique(days)
    n_days = len(unique_days)
    n_per_day = len(times) // n_days
    n_s = len(stations)

    # --- Fill tensors ---
    Xg = np.zeros((n_s, n_days, n_per_day))
    for s in stations:
        ts = df_clean[df_clean["station"] == s].sort_values("time")
        ts = resample_minute(ts, g_freq)
        ts = build_events(ts)
        ts = apply_censoring(ts)

        i = s_to_i[s]
        dep = ts["demande_obs_dep"].values.reshape(n_days, n_per_day)
        arr = ts["demande_obs_arr"].values.reshape(n_days, n_per_day)

        Xg[i] = (
            np.apply_along_axis(gaussian_nan_filter, 1, arr, g_sigma)
            - np.apply_along_axis(gaussian_nan_filter, 1, dep, g_sigma)
        )

    # --- Post-process ---
    Xg = np.nan_to_num(Xg)
    Xg = np.clip(Xg, -5, 5)

    # --- Downsample ---
    ratio = int(pd.Timedelta(t_freq) / pd.Timedelta(g_freq))
    n_s, n_d, n_t = Xg.shape
    Xg = Xg.reshape(n_s, n_d, n_t // ratio, ratio).sum(-1)

    # --- Lag + Tucker ---
    print("Calcul Tucker (cela peut prendre quelques minutes)...")
    XL = add_lag_dimension(Xg, max_lag)
    X_rec = tucker_reconstruction(XL, ranks=t_ranks)
    X_final = X_rec[..., 0].reshape(n_s, -1)

    # --- Dataframe & Export ---
    df_out = tensor_to_dataframe(X_final, stations, tensor_times)
    df_out = df_out.merge(df_clean_20, on=["station", "time"], how="left")

    df_out.to_csv(output_csv, index=False)
    print(f"✅ Reconstruction terminée : {output_csv}")
    return df_out

# Pour permettre de tester le fichier seul si besoin
if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        conf = json.load(f)
    run_reconstruction(conf)