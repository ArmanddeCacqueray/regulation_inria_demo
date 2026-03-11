#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import ast
import gurobipy as gp
from pathlib import Path
from sklearn.metrics.pairwise import haversine_distances

# Import de tes classes métier
from src.optim.planvisit import Weekplan
from src.optim.planrout import TruckRoutes
from src.optim.visualizer import TruckRoutesVisualizer

def load_optimization_data(config, mini_sample=False):
    """
    Charge les frontières et les coordonnées GPS.
    Si mini_sample=True, réduit à 5 stations vides et 5 stations pleines max.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    out_dir = Path(config["paths"]["output_dir"])
    
    frontiers = pd.read_csv(out_dir / "frontiers_strategies.csv")
    lat_lon = pd.read_csv(raw_dir / "attributs.csv")

    if mini_sample:
        print("💡 Mode Mini-Sample activé (Max 5 stations de chaque signe pour la version gratuite)")
        vides = frontiers[frontiers["sign"] == 15].sample(n=min(5, len(frontiers[frontiers["sign"] == 15])))
        pleins = frontiers[frontiers["sign"] == -15].sample(n=min(5, len(frontiers[frontiers["sign"] == -15])))
        frontiers = pd.concat([vides, pleins]).reset_index(drop=True)

    stations_all = frontiers[["station", "sign"]].copy()
    stations_all = stations_all.merge(
        lat_lon[["station_code", "latitude", "longitude"]],
        left_on="station",
        right_on="station_code",
        how="left"
    )
        # 1. Calcul de la moyenne des coordonnées existantes
    mean_lat = stations_all['latitude'].mean()
    mean_lon = stations_all['longitude'].mean()

    # 2. Remplissage des NaN avec ces valeurs
    stations_all['latitude'] = stations_all['latitude'].fillna(mean_lat)
    stations_all['longitude'] = stations_all['longitude'].fillna(mean_lon)
    coords_rad = np.radians(stations_all[["latitude", "longitude"]].values)

    distance_matrix = haversine_distances(coords_rad) * 6371 

    return frontiers, stations_all, distance_matrix

def prepare_optimization_params(frontiers, stations_all, distance_matrix, config, mini_sample=False):
    """Transforme les données brutes pour les solveurs."""
    
    def parse_frontiers_logic(cell):
        if not isinstance(cell, str): return cell
        raw_list = ast.literal_eval(cell)
        limit = 2 if mini_sample else 7
        return [list(map(int, x.strip("[]")))[:limit] for x in raw_list]

    params = {"vide": {}, "plein": {}, "routing": {}}
    nin_limit = config["params"].get("nin_limit", 50)
    
    for sens, sign_val in [("vide", 15), ("plein", -15)]:
        sub = frontiers[frontiers["sign"] == sign_val].copy()
        params[sens] = {
            "station_ids": sub["station"].tolist(),
            "strategies": {
                "down": sub["frontiere_bas"].apply(parse_frontiers_logic).tolist(),
                "up": sub["frontiere_haut"].apply(parse_frontiers_logic).tolist()
            },
            "Nin": nin_limit,
            "active_mask": (frontiers["sign"] == sign_val).values,
            "losses": 0
        }

    params["routing"]["distance_matrix"] = distance_matrix
    params["routing"]["station_ids_global"] = stations_all["station_code"].tolist()
    params["routing"]["penalty_same_type"] = config["params"].get("penalty_same_type", 5)
    
    try:
        sample_strat = params["vide"]["strategies"]["down"][0][0]
        n_days = len(sample_strat)
    except (IndexError, KeyError):
        n_days = 2 if mini_sample else 7

    dims = {
        "S_vide": range(len(params["vide"]["station_ids"])),
        "S_plein": range(len(params["plein"]["station_ids"])),
        "N": range(n_days)
    }
    return dims, params

def run_optimization(config):
    print("--- Début de l'Optimisation Logistique ---")
    out_dir = Path(config["paths"]["output_dir"])
    
    # 1. Tentative avec les paramètres complets (Grand format)
    mini_sample = False
    frontiers, stations_all, dist_matrix = load_optimization_data(config, mini_sample=mini_sample)
    dims, params = prepare_optimization_params(frontiers, stations_all, dist_matrix, config, mini_sample=mini_sample)

    try:
        #print("  Étape 1 : Calcul du plan stratégique (Weekplan)...")
        #wp = Weekplan(dims, params, verbose=False)
        #wp.solve()

        print("  Étape 2 : Résolution du routage opérationnel...")
        n_models = 3 if config.get("solve", "best") == "best" else 2
        routs = TruckRoutes(dims, params, verbose=True, nmodels=n_models, solve = config.get("solve", "best"))
        
        # On tente de résoudre le premier modèle
        routs.solve(0)
        
    except gp.GurobiError as e:
        if "size-limited license" in str(e) or "Model too large" in str(e):
            print("\n⚠️  ERREUR : Problème trop complexe pour la licence actuelle (limite 2000 variables).")
            print("🔄 Basculement automatique en mode 'Mini-Sample'... (ou alors installer une licence commerciale/recherche)")
            
            # Rechargement en mode réduit
            mini_sample = True
            frontiers, stations_all, dist_matrix = load_optimization_data(config, mini_sample=mini_sample)
            dims, params = prepare_optimization_params(frontiers, stations_all, dist_matrix, config, mini_sample=mini_sample)
            
            # On réinitialise les modèles
            wp = Weekplan(dims, params, verbose=False)
            wp.solve()
            
            n_models = 1 # En mode démo, on ne fait qu'un seul modèle pour aller vite
            routs = TruckRoutes(dims, params, verbose=False, nmodels=n_models)
            routs.solve(0)
        else:
            raise e # Relancer si c'est une autre erreur Gurobi

    # Suite de la boucle pour les modèles restants (si pas en mode mini)
    if not mini_sample:
        for m in range(1, n_models):
            print(f"    Optimisation du modèle {m+1}/{n_models}...")
            routs.solve(m)
    
    routs.to_csv(out_dir / "planning_camions_final.csv", id_model=n_models-1)
    print(f"✅ Terminé ! Fichier : {out_dir / 'planning_camions_final.csv'}")

    # 3. Visualisation
    if config.get("visualize", False):
        print("  Génération des rapports et cartes...")
        try:
            vis = TruckRoutesVisualizer(routs, stations_all)
            vis.extract_chains(m=n_models-1)
            vis.save_routes_to_txt(m=n_models-1, output_dir=out_dir)
            vis.plot_routes(m=n_models-1, output_dir=out_dir)
        except Exception as e:
            print(f"  Note : Erreur visualisation ({e})")

if __name__ == "__main__":
    config_path = Path("config.json")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent / "config.json"
        
    with open(config_path, "r") as f:
        run_optimization(json.load(f))