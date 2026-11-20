import os
import pandas as pd
from src.exceptions import DonneeInvalideError

COLONNES = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

def charger_donnees(chemin: str) -> pd.DataFrame:
    """Lit le fichier CSV et vérifie que toutes les colonnes attendues sont là.
    Reçoit : le chemin du fichier CSV.
    Renvoie : un DataFrame pandas avec les données.
    Lève l'exception FileNotFoundError si le fichier n'existe pas, et avec
    DonneeInvalideError s'il manque une colonne attendue.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    with open(chemin, encoding="utf-8") as fichier:
        donnees = pd.read_csv(fichier)

    manquantes = [c for c in COLONNES if c not in donnees.columns]
    if manquantes:
        raise DonneeInvalideError(f"Colonnes manquantes : {manquantes}")

    return donnees