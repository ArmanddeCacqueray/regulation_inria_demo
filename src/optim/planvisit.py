import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

class Weekplan:
    def __init__(self, dims, params, verbose=False, build_obj=True, nmodels=1):
        self.build_obj = build_obj
        self.init_dims = dims
        self.nmodels = nmodels
        self.p, self.dims = params, dims
        self.params = self.p
        self.verbose = verbose
        self.decision_mode = GRB.BINARY if build_obj else GRB.CONTINUOUS

        # Créer une liste de modèles, même si nmodels = 1
        self.model = [gp.Model(f"bike_rebalancing_{i}") for i in range(nmodels)]

        if not verbose:
            # Désactiver l'output pour tous les modèles
            for m in self.model:
                m.setParam("OutputFlag", 0)

        # Dictionnaires pour stocker les variables et autres données spécifiques à chaque modèle
        self.dinj = {}
        self.score_s = {}
        self.active_strat = {}
        self.obj = {}

        # Initialiser les structures pour chaque modèle
        for m in range(nmodels):
            self.dinj[m] = {}
            self.score_s[m] = {}
            self.active_strat[m] = {}
            self.obj[m] = gp.LinExpr()  # Initialiser l'objectif pour chaque modèle

        for sens in ["vide", "plein"]:
            self._build_variables(sens)
            self._build_constraints(sens)

            # Assigner l'objectif combiné si nécessaire
            if build_obj:
                self._build_objective(sens)

        # Appliquer l'objectif à chaque modèle
        if build_obj:
            for m in range(nmodels):
                self.model[m].setObjective(self.obj[m], GRB.MAXIMIZE)

    def _build_variables(self, sens):
        S, N = self.dims[f"S_{sens}"], self.dims["N"]
        strat_list = self.p[sens]["strategies"]

        # Créer des variables pour chaque modèle
        for m in range(len(self.model)):
            # Variables pour les injections, les scores et l'activation des stratégies
            self.dinj[m][sens] = self.model[m].addVars(S, N, vtype=self.decision_mode, lb=0.0, ub=1.0, name=f"dinj_{m}")
            self.score_s[m][sens] = self.model[m].addVars(S, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"score_s_{m}")
            self.active_strat[m][sens] = {"down": {}, "up": {}}

            # Variables d'activation des stratégies down
            for s, strats in enumerate(strat_list["down"]):
                for strat in range(len(strats)):
                    self.active_strat[m][sens]["down"][s, strat] = self.model[m].addVar(
                        vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"active_strat_{m}_{s}_{strat}")
                    
            # Variables de respect des contraintes up
            for s, strats in enumerate(strat_list["up"]):
                for strat in range(len(strats)):
                    self.active_strat[m][sens]["up"][s, strat] = self.model[m].addVar(
                        vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"respect_strat_{m}_{s}_{strat}")

    def _build_constraints(self, sens):
        S, N = self.dims[f"S_{sens}"], self.dims["N"]
        strat_list = self.p[sens]["strategies"]

        for m in range(len(self.model)):
            dinj = self.dinj[m][sens]
            score_s = self.score_s[m][sens]
            active_strat = self.active_strat[m][sens]
            p = self.p[sens]

            # Nombre total de stations approvisionnées
            self.model[m].addConstrs(
                (gp.quicksum(dinj[s, n] for s in S) <= p["Nin"] for n in N),
                name=f"day_limit_inj_{m}"
            )


            # Activation stratégie
            for s, strats in enumerate(strat_list["down"]):
                for strat, strat_inj in enumerate(strats):
                    for n in N:
                        self.model[m].addConstr(
                            dinj[s, n] >= strat_inj[n] * active_strat["down"][s, strat],
                            name=f"active_strat_down_{m}_{s}_{strat}_{n}"
                        )
            for s, strats in enumerate(strat_list["up"]):
                for strat, strat_inj in enumerate(strats):
                    for n in N:
                        self.model[m].addConstr(
                            dinj[s, n] <= strat_inj[n]+ (1-active_strat["up"][s, strat]),
                            name=f"active_strat_up_{m}_{s}_{strat}_{n}"
                        )
            
            # Respect strict de la borne up
            for s, strats in enumerate(strat_list["up"]):
                self.model[m].addConstr(
                    score_s[s] <= gp.quicksum(active_strat["up"][s, strat] for strat in range(len(strats))),
                    name=f"up_succes_{m}_{s}"
                )

            # Condition de succès de la borne down
            for s, strats in enumerate(strat_list["down"]):
                self.model[m].addConstr(
                    score_s[s] <= gp.quicksum(active_strat["down"][s, strat] for strat in range(len(strats))),
                    name=f"down_succes_{m}_{s}"
                )

    def _build_objective(self, sens):
        # Poids pour briser la symétrie
        S, N = self.dims[f"S_{sens}"], self.dims["N"]
        score_importance = 20
        inj_importance = 5

        sweights = np.linspace(1, 1.1, len(S))  # poids station arbitraire
        nweights = np.linspace(1, 1.1, len(N))  # poids jour -> injecter tôt

        for m in range(len(self.model)):
            dinj = self.dinj[m][sens]
            score_s = self.score_s[m][sens]

            # Objectif 1 : Maximiser le score
            for s in S:
                self.obj[m] += score_s[s] * sweights[s] * score_importance

            # Objectif 2 : Minimiser le nombre d'injections
            for s in S:
                for n in N:
                    self.obj[m] += -dinj[s, n] * sweights[s] * nweights[n] * inj_importance

    def solve(self, id_model=0):
        if id_model is not None:
            #self.model[id_model].Params.MIPGap = 0.01
            # Résoudre un modèle spécifique
            self.model[id_model].optimize()
            return self.model[id_model].status
        else:
            # Résoudre tous les modèles (ou le seul modèle)
            statuses = []
            for m in self.model:
                m.optimize()
                statuses.append(m.status)
            return statuses
        
    def to_csv(self, filename, id_model=0, threshold=0.5):
        rows = []

        for sens, direction in zip(["vide", "plein"], [15, -15]):

            station_ids = self.p[sens]["station_ids"]
            S = self.dims[f"S_{sens}"]
            N = self.dims["N"]

            for s in S:
                score = self.score_s[id_model][sens][s].X

                # reconstruire la stratégie à partir des variables dinj
                strat = []
                for n in N:
                    val = self.dinj[id_model][sens][s, n].X
                    strat.append(int(val > threshold))


                strat_string = "[" + "".join(map(str, strat)) + "]"

                rows.append({
                    "station": station_ids[s],
                    "sign": direction,
                    "strategie": strat_string,
                    "succes": score > threshold
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"CSV exporté vers {filename}")


        

