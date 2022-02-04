# I. Import des bibliothèques

import pickle # Serialiser des objets (y comporis des modeles)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import Config

# II. Création du répertoire destiné à héberger les models

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

# III. Chargement des features et lables de la base de données d'entrainement

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

# IV.Entrainement du model
rf_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=Config.RANDON_SEED)
rf_model.fit(X_train, y_train)


# V. Enregisrement du model
pickle.dump(rf_model, open(str(Config.MODELS_PATH / "model.pikle"), mode='wb'))