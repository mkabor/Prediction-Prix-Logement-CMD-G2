# I. Import des bibliothèques

from os import remove
import pickle # Serialiser des objets (y comporis des modeles)
import json 
import math

import pandas as pd
from pandas.core.algorithms import mode
from sklearn.metrics import mean_squared_error
from config import Config

# II. Chargement des features et test de la base de données test 
X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))


# III. Restauration du modèle
model = pickle.load(open(str(Config.MODELS_PATH / "model.pikle"), mode='rb'))

# IV. Calcul des métriques du modèle avec les données test
r_squared = model.score(X_test, y_test)
y_pred = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))


# IV. Sauvegarde des métiques
with open(str(Config.METRICS_FILE_PATH), mode='w') as f:
    json.dump(dict(r_squared=r_squared, rmse=rmse), f)
