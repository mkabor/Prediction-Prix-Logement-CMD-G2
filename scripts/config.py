# I. Import des bibliothèques

from pathlib import Path

# II. Construction de la classe des paramètres

class Config:
    RANDON_SEED = 28 # Seed
    TEST_SIZE = 0.3
    ASSETS_PATH = Path("./assets")
    ORIGINAL_DATASET_FILES_PATH = ASSETS_PATH / "original_data" # Dossier pour les données originales 
    DATASET_PATH = ASSETS_PATH / "data" # Dossier pour notre dataset
    FEATURES_PATH = ASSETS_PATH / "features" # Dossier pour les features 
    MODELS_PATH = ASSETS_PATH / "models" # Dossier pour les modeles
    METRICS_FILE_PATH = ASSETS_PATH / "metrics.json" # Fichiers pour les mesures