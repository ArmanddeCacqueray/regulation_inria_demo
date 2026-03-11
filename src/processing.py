#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
from typing import Dict

COLS_CLEAN = ["station", "time", "stock", "indispo", "diapasons", "not_regulated"]
COLS_20MIN = ["station", "time", "stock", "indispo", "diapasons", "not_regulated", "capacity"]

# ============================================================
# ======================= STATION CONTROL ====================
# ============================================================

def update_station_lists(df: pd.DataFrame,
                         station_col: str,
                         process_dir: Path) -> pd.DataFrame:
    """Met à jour whitelist / blacklist après filtrage dates"""
    with_path = process_dir / "withlist.csv"
    black_path = process_dir / "blacklist.csv"

    current_stations = set(df[station_col].astype(str))

    # Charger historique
    withlist = set(pd.read_csv(with_path)["station"].astype(str)) if with_path.exists() else current_stations.copy()
    blacklist = set(pd.read_csv(black_path)["station"].astype(str)) if black_path.exists() else set()

    removed = withlist - current_stations
    new_stations = current_stations - withlist
    blacklist.update(removed)
    blacklist.update(new_stations)
    withlist &= current_stations

    pd.DataFrame({"station": sorted(withlist)}).to_csv(with_path, index=False)
    pd.DataFrame({"station": sorted(blacklist)}).to_csv(black_path, index=False)

    df = df[~df[station_col].astype(str).isin(blacklist)].copy()
    return df


def read_and_filter(path: Path,
                    station_col: str,
                    time_col: str,
                    start: pd.Timestamp,
                    end: pd.Timestamp,
                    process_dir: Path) -> pd.DataFrame:
    """Lecture CSV + filtre dates + update stations"""
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    df = pd.read_csv(path, sep=sep)

    # Vérification sécurité colonnes
    if station_col not in df.columns or time_col not in df.columns:
        raise KeyError(f"Colonnes attendues manquantes : {station_col}, {time_col}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df[(df[time_col] >= start) & (df[time_col] < end + pd.Timedelta(days=1))].copy()
    if df.empty:
        return df

    df = update_station_lists(df, station_col, process_dir)
    return df


# ============================================================
# ===================== REGULATION ===========================
# ============================================================

def load_regulation(path: Path,
                    cols: Dict[str, str],
                    start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    """Charge et filtre les régulations sur la période"""
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    reg = pd.read_csv(path, sep=sep)

    # Vérification sécurité colonnes
    for c in [cols["station_pick"], cols["station_drop"], cols["date_start"], cols["date_end"]]:
        if c not in reg.columns:
            raise KeyError(f"Colonne régulation manquante : {c}")

    reg["date_debut"] = pd.to_datetime(reg[cols["date_start"]], dayfirst=True)
    reg["date_fin"] = pd.to_datetime(reg[cols["date_end"]], dayfirst=True)
    reg = reg[(reg["date_fin"] >= start) & (reg["date_debut"] <= end + pd.Timedelta(days=1))].copy()

    delta = pd.Timedelta(minutes=15)
    reg_events = pd.concat([
        reg.assign(station=reg[cols["station_pick"]], start=reg["date_debut"]-delta, end=reg["date_fin"]+delta)[["station","start","end"]],
        reg.assign(station=reg[cols["station_drop"]], start=reg["date_debut"]-delta, end=reg["date_fin"]+delta)[["station","start","end"]]
    ], ignore_index=True)

    if not reg_events.empty:
        reg_events["station"] = reg_events["station"].astype(int)
    return reg_events


# ============================================================
# ===================== TRANSFORMATIONS ======================
# ============================================================

def enforce_bounds(df, start, end):
    if df.empty:
        return df
    start_dt = start.floor("D")
    end_dt = end.floor("D") + pd.Timedelta(hours=23, minutes=59)
    outs = []

    for st, g in df.groupby("station", sort=True):
        g = g.sort_values("time").set_index("time")
        bounds = pd.DatetimeIndex([start_dt, end_dt])
        g = g.reindex(g.index.union(bounds)).sort_index().ffill().bfill()
        g["station"] = st
        g = g.reset_index().rename(columns={"index": "time"})
        outs.append(g)

    return pd.concat(outs, ignore_index=True)



def build_station_columns(df, reg, cols, start, end):
    if df.empty:
        return pd.DataFrame(columns=["station","time","stock","indispo","diapasons","not_regulated"])
    df = df.copy()
    dispo_cols = [c for c in df.columns if cols["available_pattern"] in c and cols["not_available_pattern"] not in c]
    indispo_cols = [c for c in df.columns if cols["not_available_pattern"] in c]

    df["stock"] = df[dispo_cols].fillna(0).sum(axis=1)
    df["indispo"] = df[indispo_cols].fillna(0).sum(axis=1)
    df["diapasons"] = df[cols["docks_free"]].fillna(0) + df[cols["cable_free"]].fillna(0)

    df["not_regulated"] = 1
    for _, row in reg.iterrows():
        mask = (df[cols["time"]] >= row.start) & (df[cols["time"]] <= row.end) & (df[cols["station"]] == row.station)
        df.loc[mask, "not_regulated"] = 0

    df = df.rename(columns={cols["station"]:"station", cols["time"]:"time"})
    df = df[["station","time","stock","indispo","diapasons","not_regulated"]]
    return enforce_bounds(df, start, end)


def build_stock_resampled(df_clean, freq):
    if df_clean.empty:
        return df_clean
    outs = []
    for st, g in df_clean.groupby("station", sort=True):
        g = g.sort_values("time").set_index("time")
        g = g.resample(freq).first()
        g["not_regulated"] = g["not_regulated"].fillna(1)
        g = g.ffill().reset_index()
        g["station"] = st
        outs.append(g)
    out = pd.concat(outs, ignore_index=True)
    out["capacity"] = out[["stock","indispo","diapasons"]].sum(axis=1)
    return out


# ==========================================================
# ==================== CALENDRIER ==========================
# ==========================================================

def get_week_start(date: pd.Timestamp):
    return date - pd.Timedelta(days=date.weekday())


# ==========================================================
# ==================== PIPELINE PRINCIPAL ==================
# ==========================================================

def run_processing(config: dict):

    mode = config["mode"]
    today = pd.Timestamp(config["today"])
    process_dir = Path(config["paths"]["process_dir"])
    process_dir.mkdir(exist_ok=True)

    remplissage_path = Path(config["paths"]["remplissage_file"])
    regulation_path = Path(config["paths"]["regulation_file"])

    cols_fill = config["cols_fill"]
    cols_reg = config["cols_reg"]
    freq = config["params"]["sample_freq"]

    current_week_start = get_week_start(today)
    last_week_start = current_week_start - pd.Timedelta(days=7)
    end_period = today
   
    if mode == "init":
        df_fill = read_and_filter(remplissage_path, cols_fill["station"], cols_fill["time"], last_week_start, end_period, process_dir)
        reg = load_regulation(regulation_path, cols_reg, last_week_start, end_period)
        df_clean = build_station_columns(df_fill, reg, cols_fill, last_week_start, end_period)
        df_clean_20 = build_stock_resampled(df_clean, freq)



        # split last_week / new_week
        last_week_end = current_week_start - pd.Timedelta(days=1)
        df_last = df_clean[df_clean["time"] < current_week_start].copy()
        df_new = df_clean[df_clean["time"] >= current_week_start].copy()
        df_last_20 = df_clean_20[df_clean_20["time"] < current_week_start].copy()
        df_new_20 = df_clean_20[df_clean_20["time"] >= current_week_start].copy()
        file_new, file_new_20, file_last, file_last_20 = [
            process_dir / f"CLEAN_{label}.csv" for label in ["new_week", "new_week_20min", "last_week", "last_week_20min"]
        ]

        if today.weekday() == 6:
            print("dimanche!")
            df_new.to_csv(file_last, index=False)
            df_new_20.to_csv(file_last_20, index=False)
            pd.DataFrame(columns=COLS_CLEAN).to_csv(path_new, index=False)
            pd.DataFrame(columns=COLS_20MIN).to_csv(path_new_20, index=False)
        else:

            df_last.to_csv(file_last, index=False)
            df_last_20.to_csv(file_last_20, index=False)
            df_new.to_csv(file_new, index=False)
            df_new_20.to_csv(file_new_20, index=False)

    elif mode == "rolling":
        df_day = read_and_filter(remplissage_path, cols_fill["station"], cols_fill["time"], today, today, process_dir)
        reg = load_regulation(regulation_path, cols_reg, today, today)
        df_clean = build_station_columns(df_day, reg, cols_fill, today, today)
        df_clean_20 = build_stock_resampled(df_clean, freq)

        # chargement new_week existant
        path_new = process_dir / "CLEAN_new_week.csv"
        path_new_20 = process_dir / "CLEAN_new_week_20min.csv"
        path_last = process_dir / "CLEAN_last_week.csv"
        path_last_20 = process_dir / "CLEAN_last_week_20min.csv"

        df_new = pd.read_csv(path_new, parse_dates=["time"]) if path_new.exists() else pd.DataFrame()
        df_new_20 = pd.read_csv(path_new_20, parse_dates=["time"]) if path_new_20.exists() else pd.DataFrame()

        df_new = pd.concat([df_new, df_clean], ignore_index=True).sort_values(["station","time"])
        df_new_20 = pd.concat([df_new_20, df_clean_20], ignore_index=True).sort_values(["station","time"])

        # rollover dimanche
        if today.weekday() == 6:
            print("dimanche!")
            df_new.to_csv(path_last, index=False)
            df_new_20.to_csv(path_last_20, index=False)
            pd.DataFrame(columns=COLS_CLEAN).to_csv(path_new, index=False)
            pd.DataFrame(columns=COLS_20MIN).to_csv(path_new_20, index=False)
        else:
            df_new.to_csv(path_new, index=False)
            df_new_20.to_csv(path_new_20, index=False)
    else:
        raise ValueError("mode doit être 'init' ou 'rolling'")

