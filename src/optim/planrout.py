from src.rebalancing.optim.planvisit import Weekplan
import numpy as np
import random
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import pandas as pd

DEPOT = 0
VIDE  = 1
PLEIN = 2

DEPOT = 0
VIDE  = 1
PLEIN = 2

class TruckRoutes(Weekplan):
    """
    VRP unifié avec :
    - un dépôt unique
    - un id unique par station
    - une seule variable d’arc
    - distances basées sur params["routing"]["distance_matrix"]
    """

    # =====================================================
    # Init
    # =====================================================
    def __init__(self, dims, params, verbose=False, nmodels=3, solve = "best"):
        self.solve_mode = solve
        super().__init__(dims, params, verbose=verbose, build_obj=False, nmodels=nmodels)
        self.modeldim = range(nmodels)
        self.N = self.dims["N"]          # jours
        self.C = 5                        # nombre de camions
        self.nmax = 10                    # nombre de stations vides max par tournée
        self.topk = [5, 10, 20]           # k plus proches voisins pour arcs
        self.rdmconnect = [5, 10, 10]     # arcs aléatoires pour connexité
        self.arcs_dict = {}
        self.arcs_per_day = {m: {n: defaultdict(list) for n in self.N} for m in self.modeldim}

        self._merge()     
        self._build_dist()             
        self._build_r_variables()
        self._build_r_constraints()
        self._build_r_flow()
        self._build_r_objectives()

    # =====================================================
    # Merge vide + plein + dépôt
    # =====================================================
    def _merge(self):
        """
        Crée un graphe unifié :
        - nodes : ids internes
        - type[i] : DEPOT / VIDE / PLEIN
        - global_id[i] : id station réel
        - score[m][i]
        - din[m][i,n]
        """
        self.nodes = [0]
        self.type = {0: DEPOT}
        self.global_id = {0: None}

        self.score = {m: {0: 0} for m in self.modeldim}
        self.din = {m: {(0, n): self.C for n in self.N} for m in self.modeldim}

        idx = 1
        # --- Stations VIDE ---
        for s in self.dims["S_vide"]:
            self.nodes.append(idx)
            self.type[idx] = VIDE
            self.global_id[idx] = self.params["vide"]["station_ids"][s]
            for m in self.modeldim:
                self.score[m][idx] = self.score_s[m]["vide"][s]
            for m in self.modeldim:
                for n in self.N:
                    self.din[m][idx, n] = self.dinj[m]["vide"][s, n]
            idx += 1
        # --- Stations PLEIN ---
        for s in self.dims["S_plein"]:
            self.nodes.append(idx)
            self.type[idx] = PLEIN
            self.global_id[idx] = self.params["plein"]["station_ids"][s]
            for m in self.modeldim:
                self.score[m][idx] = self.score_s[m]["plein"][s]
            for m in self.modeldim:
                for n in self.N:
                    self.din[m][idx, n] = self.dinj[m]["plein"][s, n]
            idx += 1

    # =====================================================
    # Distance via params["routing"]
    # =====================================================
    def _build_dist(self):
        dist_global = self.params["routing"]["distance_matrix"]
        global_ids = self.params["routing"]["station_ids_global"]
        penalty_same = self.params["routing"].get("penalty_same_type", 0)
        id_to_index = {gid: k for k, gid in enumerate(global_ids)}

        S = len(self.nodes)
        self.dist_mat = np.zeros((S, S))

        for i in self.nodes:
            for j in self.nodes:
                if i == 0 or j == 0:
                    self.dist_mat[i, j] = 0
                    continue
                gi = self.global_id[i]
                gj = self.global_id[j]
                ii = id_to_index[gi]
                jj = id_to_index[gj]
                d = dist_global[ii, jj]
                if self.type[i] == self.type[j]:
                    d += penalty_same
                self.dist_mat[i, j] = d

    # =====================================================
    # Variables
    # =====================================================
    def _build_r_variables(self):
        self.x = {m: {} for m in self.modeldim}
        # 1️⃣ Initialisation globale
        for m in self.modeldim:
            for n in self.N:
                for i in self.nodes:
                    for j in self.nodes:
                        vname = f"x_{i}_{j}_{n}"
                        self.x[m][i, j, n] = (
                            self.model[m].addVar(vtype=GRB.BINARY, name=vname)
                            if i == DEPOT or j == DEPOT
                            else 0
                        )
                        if i==DEPOT or j==DEPOT:
                            self.x[m][i, j, n].Partition = n +1
        # 2️⃣ Construction par jour
        for n in self.N:
            for i in self.nodes:
                if i == DEPOT: continue
                candidates_dict = {
                    "diff_type": [j for j in self.nodes if j != DEPOT and self.type[i] != self.type[j]],
                    "same_type": [j for j in self.nodes if j != DEPOT and self.type[i] == self.type[j]]
                }
                for cand_type, candidates in candidates_dict.items():
                    if not candidates: continue
                    candidates.sort(key=lambda j: self.dist_mat[i, j])
                    topk_val = min(self.topk[-1], len(candidates))
                    topk_i = candidates[:topk_val]
                    remaining = candidates[topk_val:]
                    rdmconnect_i = random.sample(remaining, min(len(remaining), self.rdmconnect[-1])) if remaining else []
                    for list_i, model_connectivity in zip([topk_i, rdmconnect_i], [self.topk, self.rdmconnect]):
                        for nj, j in enumerate(list_i):
                            vname = f"x_{i}_{j}_{n}"
                            for m in self.modeldim:
                                if nj <= model_connectivity[m]:
                                    self.x[m][i, j, n] = self.model[m].addVar(vtype=GRB.BINARY, name=vname)
                                    self.x[m][i, j, n].Partition = n + 1
    # =====================================================
    # Contraintes
    # =====================================================
    def _build_r_constraints(self):
        for m in self.modeldim:
            for n in self.N:
                for i in self.nodes:
                    # conservation de flux
                    self.model[m].addConstr(
                        gp.quicksum(self.x[m][i, j, n] for j in self.nodes)
                        ==
                        gp.quicksum(self.x[m][j, i, n] for j in self.nodes),
                        name=f"flow_{i}_{n}"
                    )
                    # activation / visite
                    self.model[m].addConstr(
                        gp.quicksum(self.x[m][i, j, n] for j in self.nodes)
                        == self.din[m][i, n],
                        name=f"visit_{i}_{n}"
                    )

    # =====================================================
    # MTZ (anti-sous-tours)
    # =====================================================
    def _build_r_flow(self):
                # Flots anti-sous-tours séparés
        self.f_vide = {m: {} for m in self.modeldim}
        self.f_plein = {m: {} for m in self.modeldim}

        for m in self.modeldim:
            for (i, j, n), x_var in self.x[m].items():
                if isinstance(x_var, gp.Var):
                    self.f_vide[m][i, j, n] = self.model[m].addVar(lb=0, ub=self.nmax, vtype=GRB.CONTINUOUS, name=f"f_vide_{i}_{j}_{n}")
                    self.f_plein[m][i, j, n] = self.model[m].addVar(lb=0, ub=self.nmax, vtype=GRB.CONTINUOUS, name=f"f_plein_{i}_{j}_{n}")
                else:
                    self.f_vide[m][i, j, n] = 0
                    self.f_plein[m][i, j, n] = 0

        M = self.nmax
        for m in self.modeldim:
            for n in self.N:
                for i in self.nodes:
                    for j in self.nodes:
                        var = self.x[m].get((i,j,n), None)
                        if isinstance(var, gp.Var):
                            # flot vide
                            f_v = self.f_vide[m].get((i,j,n), 0)
                            if isinstance(f_v, gp.Var):
                                self.model[m].addConstr(f_v <= M * var)
                            # flot plein
                            f_p = self.f_plein[m].get((i,j,n), 0)
                            if isinstance(f_p, gp.Var):
                                self.model[m].addConstr(f_p <= M * var)
                                
                # conservation des flots séparés
                for i in self.nodes:
                    if i == DEPOT: continue
                    # flux vide
                    incoming_v = gp.quicksum(self.f_vide[m][j,i,n] for j in self.nodes)
                    outgoing_v = gp.quicksum(self.f_vide[m][i,j,n] for j in self.nodes)
                    is_type_vide = 1 if self.type[i]==VIDE else 0
                    self.model[m].addConstr(
                        incoming_v - outgoing_v >= self.din[m][i,n]*is_type_vide,
                        name=f"cons_flow_vide_{i}_{n}"
                    )
                    # flux plein
                    incoming_p = gp.quicksum(self.f_plein[m][j,i,n] for j in self.nodes)
                    outgoing_p = gp.quicksum(self.f_plein[m][i,j,n] for j in self.nodes)
                    is_type_plein = 1 if self.type[i]==PLEIN else 0
                    self.model[m].addConstr(
                        incoming_p - outgoing_p >= self.din[m][i,n]*is_type_plein,
                        name=f"cons_flow_plein_{i}_{n}"
                    )


    # =====================================================
    # Objectifs
    # =====================================================
    def _build_r_objectives(self):
        for m in self.modeldim:
            obj_score = gp.quicksum(self.score[m][i] for i in self.nodes if i != DEPOT)
            obj_visits = gp.quicksum(
                self.din[m][i,n] for i in self.nodes for n in self.N if i != DEPOT
            ) + gp.quicksum(
                self.x[m][i,j,n]*self.dist_mat[i,j] for i in self.nodes for j in self.nodes for n in self.N
                if self.type[i]==self.type[j]
            )
            obj_dist = gp.quicksum(
                self.x[m][i,j,n]*self.dist_mat[i,j] for i in self.nodes for j in self.nodes for n in self.N
            )
            self.model[m].setObjective(10*obj_score - obj_visits - 0.3*obj_dist)
            self.model[m].ModelSense = GRB.MAXIMIZE

    # =====================================================
    # Solve / post-traitement
    # =====================================================
    def solve(self, m, time_limit=60):
        self.model[m].Params.Method = 3
        if m==0: time_limit = (120 if self.solve_mode == "best" else 60)
        if m>=1:
            for (i,j,n), var in self.x[m-1].items():
                var2 = self.x[m].get((i,j,n))
                if not isinstance(var2, gp.Var): continue
                value = var.X if isinstance(var, gp.Var) else var
                var2.setAttr('Start', round(value))
        self.model[m].setParam('TimeLimit', time_limit)
        self.model[m].optimize()
        print("============\nFINITION:")
        self.finition(m)
        print("============")
        self.finition_paires(m)
        # extraction finale
        self.arcs_dict[m] = {k: (v.X if isinstance(v, gp.Var) else 0) for k,v in self.x[m].items()}
        return self.model[m].Status

    # LNS par jour
    def finition(self, m, time_per_day=10, n_iter=1):
        for it in range(n_iter):
            for n_focus in self.N:
                frozen_vars = []
                for (i,j,n), var in self.x[m].items():
                    if not isinstance(var, gp.Var): continue
                    if n != n_focus:
                        var.LB = var.X
                        var.UB = var.X
                        frozen_vars.append(var)
                    else:
                        var.LB, var.UB = 0, 1
                self.model[m].setParam('TimeLimit', time_per_day)
                self.model[m].optimize()
                for var in frozen_vars: var.LB, var.UB = 0,1
        self.model[m].optimize()

    # LNS sur paires de jours
    def finition_paires(self, m, time_per_pair=15, n_pairs=5):
        if self.solve_mode == "fast": n_pairs = 3
        for it in range(n_pairs):
            pair_days = random.sample(self.N, 2)
            frozen_vars = []
            for (i,j,n), var in self.x[m].items():
                if not isinstance(var, gp.Var): continue
                if n not in pair_days:
                    var.LB = var.X
                    var.UB = var.X
                    frozen_vars.append(var)
                else:
                    var.LB, var.UB = 0,1
            self.model[m].setParam('TimeLimit', time_per_pair)
            self.model[m].setParam('MIPFocus',1)
            self.model[m].optimize()
            for var in frozen_vars: var.LB, var.UB = 0,1
