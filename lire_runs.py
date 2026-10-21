
import mlflow
import pandas as pd

# 1. Récupérer les expériences de l'expérience "churn-prediction"
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# 2. Sélectionner les colonnes à afficher
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

# 3. Afficher les résultats des expériences
print(runs[colonnes].to_string(index=False))

# 4. Identifier le meilleur modèle
if not runs.empty:
    meilleur = runs.iloc[0]

    # Afficher le nom et l'identifiant du meilleur run
    print("\nMeilleur run :", meilleur["tags.mlflow.runName"])
    print("ID du run :", meilleur["run_id"][:8], "...")
else:
    print("Aucune expérience trouvée.")
