#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from logging import config
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from datetime import datetime


# ======================
# ===== CORE LOGIC =====
# ======================

def stabilize(x, ref, sign_ref, reset_mask=None, band=3):
    """Stabilisation vectorisée des stocks par rapport à l'historique."""
    y = x.copy()
    delta = x - ref

    # 1) Entrée bande neutre
    inside = (np.abs(delta) <= band) | (sign_ref == 0)
    x[inside] = ref[inside]
    sign_ref[inside] = 0

    # 2) Violation de l'invariant
    active = sign_ref != 0
    bad = active & (np.sign(delta) != sign_ref)
    x[bad] = ref[bad]
    sign_ref[bad] = 0

    # 3) Reset invariant si régulation
    if reset_mask is not None:
        upd = reset_mask
        x[upd] = y[upd]
        sign_ref[upd] = np.sign(x[upd] - ref[upd])

    return x, sign_ref

def _simulate(day_reg, start, cap, demand, apply_tol, metropole_h, 
              day_link=0, stock_hist=None, reg_hist=None):
    """Simulation de l'impact des stratégies sur le stock."""
    N_sim, N_day, N_step = demand.shape
    cumul = np.zeros((N_sim, N_day, N_step + day_link))
    mask  = np.ones((N_sim, N_day), bool)

    prev = start.copy()
    sign_ref = None
    if stock_hist is not None:
        sign_ref = np.sign(prev - stock_hist[:, 0, 0])

    for d in range(N_day):
        # Régulation journalière candidate
        test = prev + day_reg[:, d]
        mask[:, d] = (test >= -apply_tol) & (test <= cap + apply_tol)
        x = np.clip(test, 0, cap)

        if sign_ref is not None:
            reset = (day_reg[:, d] != 0) | reg_hist[:, d, 0]
            x, sign_ref = stabilize(x, stock_hist[:, d, 0], sign_ref, reset_mask=reset)

        cumul[:, d, 0] = x

        for t in range(N_step):
            x = np.clip(x + demand[:, d, t], 0, cap)
            if sign_ref is not None:
                reset = reg_hist[:, d, t] if reg_hist is not None else None
                x, sign_ref = stabilize(x, stock_hist[:, d, t], sign_ref, reset_mask=reset)
            cumul[:, d, t+1] = x
        prev = x

    return cumul, mask

def int_to_binary_matrix(indices, n_bits):
    # La magie : on déplace les bits et on utilise l'opérateur AND (&)
    return (indices[:, None] & (1 << np.arange(n_bits)[::-1]) > 0).astype(int)

# ======================
# ===== WRAPPERS =======
# ======================

def load_data_clean(path, config):
    """Charge les données en filtrant la blacklist."""
    in_dir = Path(config["paths"]["process_dir"])
    try:
        blacklist = set(pd.read_csv(in_dir / "blacklist.csv")["station"].unique())
    except FileNotFoundError:
        blacklist = set()

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])  # conversion permanente
    df = df[df["time"] < pd.to_datetime(config["today"]) + pd.Timedelta(days=1)]
    df = df[~df["station"].isin(blacklist)]
    stations = df["station"].unique()
    return df, stations, len(stations)

def run_evaluation(config):
    """Pipeline principal d'évaluation."""
    print("--- Début Évaluation des Stratégies ---")
    
    # Convertir today en datetime
    cur_day = datetime.strptime(config["today"], "%Y-%m-%d")
    
    # Obtenir l'indice du jour (0 = lundi, 6 = dimanche)
    n_past_day = (cur_day.weekday() + 1) %7  
    n_day_total = 7
    n_futur_day = n_day_total - n_past_day
    
    n_per_h = 3 # 20min
    n_per_day = 24 * n_per_h
    metropole_h = np.array(config["hours"]) * n_per_h
    apply_tol = config["params"].get("apply_tol", 4)
    reg_candidates = np.array([15, -15])
    
    in_dir = Path(config["paths"]["process_dir"]) 
    out_dir = Path(config["paths"]["output_dir"])
    
    # Chemins
    passif_file = in_dir / "CLEAN_new_week_20min.csv"
    forecast_file = out_dir / "RECONSTRUCTION_FINAL.csv"
    output_csv = out_dir / "evaluated_strategies.csv"

    # 1. Chargement Passif
    past_sum_hours = None
    if n_past_day > 0:
        df_p, stations, n_s = load_data_clean(passif_file, config)
        stocks = df_p['stock'].values.reshape(n_s, -1, n_per_day)
        past_sum_hours = stocks[:, :n_past_day, metropole_h].sum(axis=1)
        start = stocks[:, -1, -2*n_per_h] # 2h avant la fin
        capacite = df_p.groupby("station")["capacity"].max().values
    
    # 2. Chargement Forecast
    df_f, stations, n_s = load_data_clean(forecast_file, config)
    if n_past_day == 0:
        stocks_f = df_f['stock'].values.reshape(n_s, n_day_total, n_per_day)
        start = stocks_f[:, 0, 0] # Début de semaine
        capacite = df_f.groupby("station")["capacity"].max().values

    # Filtrer le futur uniquement
    df_future = df_f.groupby("station").apply(lambda x: x.iloc[n_past_day*n_per_day:]).reset_index(drop=True)
    demand = df_future["demande_latente"].values.reshape(n_s, n_futur_day, n_per_day)
    stock_h = df_future["stock"].values.reshape(n_s, n_futur_day, n_per_day)
    reg_h = (1 - df_future["not_regulated"].values.reshape(n_s, n_futur_day, n_per_day)) >= 0.1

    # 3. Simulation
    strat_idx = np.arange(2**n_futur_day)
    strat_bits = int_to_binary_matrix(strat_idx, n_futur_day)
    n_strat = len(strat_bits)
    results = []

    print(f"Simulation de {n_strat} stratégies pour {n_s} stations...")
    for sign in reg_candidates:
        regs = strat_bits * sign
        for s, name in enumerate(stations):
            cumul, mask = _simulate(
                regs, 
                np.full(n_strat, start[s]), 
                np.full(n_strat, capacite[s]),
                np.repeat(demand[s][None, ...], n_strat, 0),
                apply_tol, metropole_h, day_link=1,
                stock_hist=np.repeat(stock_h[s][None, ...], n_strat, 0),
                reg_hist=np.repeat(reg_h[s][None, ...], n_strat, 0)
            )
            
            sums_future = cumul[:, :, metropole_h].sum(axis=1)
            past_s = past_sum_hours[s] if past_sum_hours is not None else 0
            week_avg = (past_s + sums_future) / n_day_total
            
            df_res = pd.DataFrame({
                "station": name,
                "strategy_idx": strat_idx,
                "strategy_bits": ["[" + "".join(map(str, b)) + "]" for b in strat_bits],
                "sign": sign,
                "applyable": mask.all(axis=1),
                "min_ratio": week_avg.min(axis=1) / capacite[s],
                "max_ratio": week_avg.max(axis=1) / capacite[s]
            })
            results.append(df_res)

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Évaluation terminée. Exporté vers {output_csv}")
    return final_df

if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        run_evaluation(json.load(f))