# I. Import des bibliothèques

import pickle # Serialiser des objets (y comporis des modeles)

import pandas as pd
from sklearn import linear_model

from config import Config

# II. Création du répertoire destiné à héberger les models

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

# III. Chargement des features et lables de la base de données d'entrainement

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

# IV.Entrainement du model
model = linear_model.LinearRegression()
model.fit(X_train, y_train.to_numpy().ravel())


# V. Enregisrement du model
pickle.dump(model, open(str(Config.MODELS_PATH / "model.pikle"), mode='wb'))