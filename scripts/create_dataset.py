

# I. Import des bibliothèques utiles pour la création de nos datasets

from pathlib import Path
from zipfile import ZipFile
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt


from config import Config



# II. Creation du dossier data dont on a besoin pour l'importation des fichiers de données.

Config.ORIGINAL_DATASET_FILES_PATH.mkdir(parents=True, exist_ok=True)
# Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)


# III. Telecharge des fichieres de données depuis kaggle

# III.1. Connexion
os.environ['KAGGLE_USERNAME'] = "kaboremoumini" # username from the json file
os.environ['KAGGLE_KEY'] = "782b6cd69c18a00fb6ffda35637bb5eb" # key from the json file

# III.2. Indexation du répertoire destiné à héberger les données
os.chdir(Config.ORIGINAL_DATASET_FILES_PATH)

# III.3. Téchargement des données
os.system("kaggle competitions download -c house-prices-advanced-regression-techniques")

# III.4. Dézipage du Téchargement des données du point II.3.
# os.system("unzip house-prices-advanced-regression-techniques.zip")
zf = ZipFile("house-prices-advanced-regression-techniques.zip", "r")
zf.extractall()
zf.close()

# shutil.unpack_archive("house-prices-advanced-regression-techniques.zip, ".")


# IV. Lecture des données train.csv et test.csv depuis le répertoire data hébergeant les données téléchargées et dezipées 

df_tain = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# V. Fusion des deux bases pour une uniformisation du traitement des variables catégorielles
df_train_test = pd.concat([df_tain, df_test], ignore_index=True)


# VI. Traitement des valeurs manquantes

# Strategie de traitement utilisee:
#    1. Utilisation de la distribution triangulaire pour l'imputation des valeurs numériques ou quantitatives
#    2. Utilisation du mode pour l'imputation des variables catégorielles


# VI.1. Définition de la fonction de distribution triangulaire pour l'imputation des variables quantitatives
def dist_Triang(df, col_name):
    minimum = df[col_name].min()
    maximum = df[col_name].max()
    mode = df[col_name].mode().values[0]
    
    np.random.seed(Config.RANDON_SEED)
    U=np.random.uniform(0, 1)
    F_C = (mode - minimum)/(maximum - minimum)
    if(0<U and U<F_C):
        X = minimum + sqrt((U*(maximum - minimum)*(mode - minimum)))
    if(F_C<=U and U<1):
        X = maximum - sqrt(((1-U)*(maximum - minimum)*(maximum - mode)))
    return X

# VI.2 Imputation des valeurs manquantes de df_train_test suivant la strategie definie plus haut 
for c in df_train_test:
    if c != "SalePrice":
        if df_train_test[c].dtype.kind == 'O':
            df_train_test[c].fillna(df_train_test[c].mode().values[0], inplace=True)
        else:
            df_train_test[c].fillna(dist_Triang(df_train_test, c), inplace=True)

# VII. Dichotomisation des variables catégorielles de df_train_test
df_train_test_treated = pd.get_dummies(df_train_test)


# VIII. Séparation des deux bases de données df_train et df_test après le traiment (Correction des valeurs manquantes et dichotomisation)
# VIII.1. Séparartion des deux bases après le traitement
train_treated = df_train_test_treated.loc[:df_tain.index.values[-1]]
test_treated = df_train_test_treated.loc[df_tain.index.values[-1]+1:]


# VIII.2. Exclusion de la colonne SalePrice de la base de données test traitée
test_treated = test_treated[[c for c in test_treated.columns if c!="SalePrice"]] # Sélection de la base de donnée test_treated sans la colonne SalePrice


# IX. Spliting de la base de données d'entrainement traité en 70% - 30%

train_treated_train, train_treated_test = train_test_split(
    train_treated, test_size=Config.TEST_SIZE, 
    random_state=Config.RANDON_SEED
)

# X. Sauvegarde des bases de données traitées.
# X.1. Création du répertoire destiné à hébergé les données traitées
path_dataset = os.getcwd().replace('\original_data', '')
Path(os.path.join(path_dataset, "data")).mkdir(parents=True, exist_ok=True)

# X.2. Sauvegarde des bases de données traitées
train_treated.to_csv(os.path.join(path_dataset, "data","train_treated.csv"), index=False)
test_treated.to_csv(os.path.join(path_dataset, "data","test_treated.csv"), index=False)
train_treated_train.to_csv(os.path.join(path_dataset, "data","train_treated_train.csv"), index=False)
train_treated_test.to_csv(os.path.join(path_dataset, "data","train_treated_test.csv"), index=False)