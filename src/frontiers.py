#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple, Set, List
from pathlib import Path

# =========================
# ===== UTILITAIRES =======
# =========================

def parse_bits(s: str) -> Tuple[int, ...]:
    """Convert '[0101]' → (0,1,0,1)"""
    return tuple(map(int, s.strip("[]")))

def build_partial_orders(strats: List[str]) -> Tuple[Dict, Dict]:
    """
    Précalcule les relations d'ordre partiel entre les stratégies.
    Indispensable pour trouver les frontières de Pareto.
    """
    bits = {s: parse_bits(s) for s in strats}
    INF: Dict[str, Set[str]] = {}
    SUP: Dict[str, Set[str]] = {}

    for s, b in bits.items():
        inf = set()
        sup = set()
        for t, bt in bits.items():
            if t == s: continue
            # Si bt est inclus dans b (bt <= b partout)
            if all(x <= y for x, y in zip(bt, b)):
                inf.add(t)
            # Si bt inclut b (bt >= b partout)
            if all(x >= y for x, y in zip(bt, b)):
                sup.add(t)
        INF[s] = inf
        SUP[s] = sup
    return INF, SUP

def compute_frontiers_group(
    group: pd.DataFrame,
    INF: Dict[str, Set[str]],
    SUP: Dict[str, Set[str]],
) -> pd.Series:
    """Calcule les stratégies minimales et maximales d'un groupe (station/signe)."""
    goods = set(group["strategy_bits"])
    
    # Bas : Personne de "bon" n'est plus petit que moi
    low = [s for s in goods if not (INF[s] & goods)]
    # Haut : Personne de "bon" n'est plus grand que moi
    high = [s for s in goods if not (SUP[s] & goods)]

    return pd.Series({
        "frontiere_bas": low,
        "frontiere_haut": high
    })

# =========================
# ===== PIPELINE ==========
# =========================

def run_frontiers(config: dict):
    """
    Exécute le filtrage des frontières de Pareto.
    """
    print("--- Début Filtrage des Frontières ---")
    
    out_dir = Path(config["paths"]["output_dir"])
    input_file = out_dir / "evaluated_strategies.csv"
    output_file = out_dir / "frontiers_strategies.csv"

    # Récupération des seuils depuis la config (ou valeurs par défaut)
    c_vide = config.get("thresholds", {}).get("critere_vide", 0.22)
    c_plein = config.get("thresholds", {}).get("critere_plein", 0.66)
    idx_null_strat = 0

    # 1. Load
    if not input_file.exists():
        raise FileNotFoundError(f"Fichier d'évaluation introuvable : {input_file}")
    
    df = pd.read_csv(input_file)

    # 2. Filtrage des "Bonnes" stratégies
    # Une stratégie est bonne si elle respecte la capacité et les seuils de service
    df["good_strat"] = (
        df["applyable"]
        & (df["max_ratio"] >= c_vide)
        & (df["min_ratio"] <= c_plein)
    )

    # 3. Suppression des stations "Autopass"
    # Si la stratégie "ne rien faire" (0000...) fonctionne déjà, 
    # on n'a pas besoin d'envoyer un camion.
    autopass = set(
        df.loc[(df["strategy_idx"] == idx_null_strat) & (df["good_strat"]), "station"].unique()
    )
    
    if autopass:
        print(f"  Info : {len(autopass)} stations n'ont pas besoin de régulation (Auto-pass).")
        df = df[~df["station"].isin(autopass)]

    # 4. Garder uniquement les bonnes stratégies
    df = df[df["good_strat"]].copy()

    if df.empty:
        print("⚠️ Attention : Aucune stratégie valide trouvée avec les seuils actuels.")
        return None

    # 5. Calcul des frontières
    print("  Calcul de l'ordre partiel...")
    all_strats = df["strategy_bits"].unique().tolist()
    INF, SUP = build_partial_orders(all_strats)

    print("  Extraction des frontières de Pareto par station...")
    frontiers = (
        df.groupby(["station", "sign"], sort=False)
          .apply(compute_frontiers_group, INF=INF, SUP=SUP)
          .reset_index()
    )

    # 6. Save
    frontiers.to_csv(output_file, index=False)
    print(f"✅ Frontières sauvegardées → {output_file}")
    print(f"   Stations à réguler : {frontiers['station'].nunique()}")
    
    return frontiers

if __name__ == "__main__":
    import json
    with open("config.json", "r") as f:
        run_frontiers(json.load(f))