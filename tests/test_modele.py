import pandas as pd
import pytest

from src.exceptions import DonneeInvalideError
from src.modele import encoder_categories, separer_donnees


def creer_petit_jeu() -> pd.DataFrame:
    """Crée un petit DataFrame de 10 lignes qui imite le jeu d'assurance.
    Ne reçoit rien. Renvoie : le DataFrame avec les 7 colonnes attendues.
    """
    regions = ["northeast", "northwest", "southeast", "southwest"]
    return pd.DataFrame({
        "age": [20 + i for i in range(10)],
        "sex": ["male" if i % 2 else "female" for i in range(10)],
        "bmi": [25.0 + i for i in range(10)],
        "children": [i % 3 for i in range(10)],
        "smoker": ["yes" if i % 3 == 0 else "no" for i in range(10)],
        "region": [regions[i % 4] for i in range(10)],
        "charges": [1000.0 * (i + 1) for i in range(10)],
    })


def test_encodage() -> None:
    """Les colonnes catégorielles deviennent des colonnes numériques.
    Ne reçoit rien. Renvoie : rien, échoue si une vérification est fausse.
    """
    entrees = creer_petit_jeu().drop(columns=["charges"])
    resultat = encoder_categories(entrees)
    assert "sex" not in resultat.columns
    assert "sex_male" in resultat.columns
    assert "smoker_yes" in resultat.columns
    assert all(resultat.dtypes != object)