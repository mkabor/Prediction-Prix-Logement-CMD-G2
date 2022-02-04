# I. Import des bibliothèques

import pandas as pd
from config import Config


# II. Créaton du repertoire destiné à héberger les features

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

# III. Chargement des base bases de données d'entrainement et de test traité

train = pd.read_csv(str(Config.DATASET_PATH / "train_treated_train.csv"))
test = pd.read_csv(str(Config.DATASET_PATH / "train_treated_test.csv"))

# IV. Sélection des features et des labels des bases de données d'entrainement et de test traité tout en excluant la colonne Id de la base des features
train_features = train[[c for c in train.columns if c not in ["Id", "SalePrice"]]]
# train_labels = train["SalePrice"]
test_features = test[[c for c in test.columns if c not in ["Id", "SalePrice"]]]
# test_labels = test["SalePrice"]

# V. Sauvegarde des features et des labels
train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=False)
train.SalePrice.to_csv(str(Config.FEATURES_PATH / "train_labels.csv"), index=False)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=False)
test.SalePrice.to_csv(str(Config.FEATURES_PATH / "test_labels.csv"), index=False)