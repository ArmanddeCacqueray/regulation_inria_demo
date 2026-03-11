import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Constantes pour la clarté
DEPOT = 0
VIDE  = 1
PLEIN = 2

class TruckRoutesVisualizer:
    """
    Classe utilitaire pour visualiser et sauvegarder les tournées.
    """

    def __init__(self, truck_routes, stations_df):
        self.tr = truck_routes
        self.stations_all = stations_df
        # Palette de couleurs pour les camions
        self.truck_colors = plt.cm.get_cmap('tab10', max(10, self.tr.C))
        self._init_positions()

    def _init_positions(self):
        """Associe chaque nœud de l'optimisation à ses coordonnées GPS."""
        self.tr.pos = {}
        lons, lats = [], []

        for i in self.tr.nodes:
            if i == 0: continue # Le dépôt est géré après
            idx_global = self.tr.global_id[i]
            row = self.stations_all[self.stations_all["station_code"] == idx_global]
            if not row.empty:
                lon, lat = row["longitude"].values[0], row["latitude"].values[0]
                self.tr.pos[i] = (lon, lat)
                lons.append(lon)
                lats.append(lat)
            else:
                self.tr.pos[i] = (0, 0)

        # Placement du dépôt au barycentre pour une carte centrée
        if lons and lats:
            self.tr.pos[0] = (np.mean(lons), np.mean(lats))
        else:
            self.tr.pos[0] = (2.3488, 48.8534) # Défaut Paris

    def _get_ordered_route(self, m, n, k):
        """
        Reconstruit la séquence ordonnée des labels (ex: ['D', 'V102', 'P504', 'D'])
        à partir des arcs activés.
        """
        arcs = self.tr.arcs_per_day[m][n].get(k, [])
        if not arcs: return []
        
        route_labels = ["D"]
        current_node = 0
        temp_arcs = list(arcs)
        
        while temp_arcs:
            # On cherche l'arc qui part du nœud actuel
            found = False
            for i, (start_node, end_node) in enumerate(temp_arcs):
                if start_node == current_node:
                    current_node = end_node
                    # Création du label
                    if end_node == 0:
                        label = "D"
                    else:
                        prefix = "V" if self.tr.type[end_node] == VIDE else "P"
                        label = f"{prefix}{self.tr.global_id[end_node]}"
                    
                    route_labels.append(label)
                    temp_arcs.pop(i)
                    found = True
                    break
            
            if not found or current_node == 0:
                break
        return route_labels

    def extract_chains(self, m, tol=0.5):
        """Organise les arcs bruts du solveur par jour et par camion."""
        for n in self.tr.N:
            # On récupère tous les arcs activés pour ce jour
            arcs_day = []
            for i in self.tr.nodes:
                for j in self.tr.nodes:
                    if i == j: continue
                    val = self.tr.arcs_dict[m].get((i, j, n), 0)
                    if val > tol:
                        # On suppose ici que le solveur stocke quel camion fait quel arc
                        # Si ton modèle ne stocke pas k dans arcs_dict, on utilise une logique simple :
                        arcs_day.append((i, j))

            # Dispatching simplifié par camion (1 chaîne par camion)
            # Note : Adapter si ton modèle gère explicitement l'index k dans les variables
            used_arcs = set()
            for k in range(self.tr.C):
                chain = []
                curr = 0
                while True:
                    next_node = None
                    for (i, j) in arcs_day:
                        if i == curr and (i, j) not in used_arcs:
                            next_node = j
                            used_arcs.add((i, j))
                            break
                    if next_node is None: break
                    chain.append((curr, next_node))
                    curr = next_node
                    if curr == 0: break
                self.tr.arcs_per_day[m][n][k] = chain

    def print_routes(self, m):
        """Affiche les tournées proprement dans la console."""
        for n in self.tr.N:
            print(f"\n=== TOURNÉES JOUR {n} ===")
            for k in range(self.tr.C):
                route = self._get_ordered_route(m, n, k)
                if len(route) > 1:
                    print(f"Camion {k}: {' -> '.join(route)}")

    def save_routes_to_txt(self, m, output_dir="data/outputs"):
        """Sauvegarde les tournées dans un fichier texte (Correction du bug chinois)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_to_save = out_path / "plan_regulation.txt"
        
        with open(file_to_save, "w", encoding="utf-8") as f:
            for n in self.tr.N:
                f.write(f"\n=== TOURNÉES JOUR {n} ===\n")
                for k in range(self.tr.C):
                    route = self._get_ordered_route(m, n, k)
                    if len(route) > 1:
                        f.write(f"Camion {k}: {' -> '.join(route)}\n")
        print(f"    -> Plan textuel sauvegardé : {file_to_save}")

    def plot_routes(self, m, day=None, output_dir="data/outputs"):
        """Génère les cartes Matplotlib."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        days_to_plot = self.tr.N if day is None else [day]

        for n in days_to_plot:
            plt.figure(figsize=(12, 8))
            
            # 1. Dessiner les stations
            for node, pos in self.tr.pos.items():
                if node == 0:
                    plt.scatter(pos[0], pos[1], c='blue', marker='s', s=200, label="Dépôt", zorder=5)
                else:
                    c = 'green' if self.tr.type[node] == VIDE else 'red'
                    plt.scatter(pos[0], pos[1], c=c, s=100, edgecolors='black', zorder=4)

            # 2. Dessiner les arcs par camion
            for k in range(self.tr.C):
                arcs = self.tr.arcs_per_day[m][n].get(k, [])
                if not arcs: continue
                color = self.truck_colors(k)
                
                for i, j in arcs:
                    p1, p2 = self.tr.pos[i], self.tr.pos[j]
                    plt.annotate("", xy=p2, xytext=p1,
                                 arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.8))

            plt.title(f"Plan de Régulation - Jour {n}")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            
            # Sauvegarde
            img_path = out_path / f"tournee_jour_{n}.png"
            plt.savefig(img_path)
            print(f"    -> Carte sauvegardée : {img_path}")
            plt.close() # Libère la mémoire