
# ===== 1. Importer les bibliothèques =====

from typing import Literal
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Créer l'application =====

app = FastAPI(title="API Churn")


# ===== 3. Charger le modèle champion UNE SEULE FOIS au démarrage =====

# Définir l'emplacement de MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Charger le modèle champion depuis le Model Registry
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Décrire les données attendues =====

class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Route /health : vérifier que l'API fonctionne =====

@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Route /predict : effectuer une prédiction =====

@app.post("/predict")
def predict(client: Client):

    # a) Transformer les données du client en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # b) Transformer la variable catégorielle en colonnes binaires (0/1)
    ligne = pd.get_dummies(ligne)

    # c) Garder les colonnes utilisées pendant l'entraînement,
    # dans le même ordre. Les colonnes absentes prennent la valeur 0.
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Effectuer la prédiction
    # 1 = le client part
    # 0 = le client reste
    resultat = int(model.predict(ligne)[0])

    # e) Retourner le résultat au format JSON
    return {
        "churn": resultat,
        "message": "Le client va partir" if resultat == 1
                   else "Le client reste"
    }
