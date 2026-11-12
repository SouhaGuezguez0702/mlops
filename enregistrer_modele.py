
# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier


# ===== 2. Se connecter au serveur MLflow lancé par Docker Compose =====
# Depuis notre poste, le serveur est accessible sur localhost:5000
mlflow.set_tracking_uri("http://localhost:5000")

# Créer ou sélectionner l'expérience
mlflow.set_experiment("churn-docker")


# ===== 3. Lire et préparer les données =====
df = pd.read_csv("churn_data.csv")

# Transformer la variable catégorielle "abonnement" en colonnes 0/1
X = pd.get_dummies(
    df.drop(columns=["churn"]),
    columns=["abonnement"]
)

y = df["churn"]


# ===== 4. Entraîner le modèle =====
modele = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modele.fit(X, y)


# ===== 5. Enregistrer le modèle dans le Registry MLflow =====
# registered_model_name : nom du modèle dans le catalogue
# serialization_format : format de sérialisation du modèle

with mlflow.start_run():

    mlflow.sklearn.log_model(
        sk_model=modele,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle",
    )


# ===== 6. Poser l'alias "champion" sur la version créée =====
client = MlflowClient()

# Récupérer les versions du modèle enregistré
versions = client.search_model_versions(
    "name='churn-model'"
)

# Récupérer la version la plus récente
version = max(
    versions,
    key=lambda v: int(v.version)
).version

# Associer l'alias champion à cette version
client.set_registered_model_alias(
    "churn-model",
    "champion",
    version
)

print(
    f"Version {version} de churn-model "
    "enregistrée avec l'alias champion"
)
