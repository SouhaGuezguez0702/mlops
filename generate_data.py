import pandas as pd
import numpy as np

# Fixer la graine aléatoire pour garantir la reproductibilité
np.random.seed(42)

# Nombre de clients fictifs à générer
n = 500

# Création du jeu de données
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n)
})

# Création de la variable cible : churn
# Un client est considéré comme à risque s'il :
# - a une consommation inférieure à 30
# - ET une ancienneté inférieure à 2 ans

data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)

# Sauvegarde des données dans un fichier CSV
data.to_csv("churn_data.csv", index=False)

# Affichage des premières lignes
print(data.head())

# Affichage du taux de churn
print(f"Taux de churn : {data['churn'].mean():.1%}")