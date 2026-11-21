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
    
def test_tailles() -> None:
    """Avec 10 lignes, le test en reçoit 2 et l'entraînement 8.
    Ne reçoit rien. Renvoie : rien, échoue si les tailles sont fausses.
    """
    x_entr, x_test, y_entr, y_test = separer_donnees(creer_petit_jeu())
    assert len(x_entr) == 8 and len(y_entr) == 8
    assert len(x_test) == 2 and len(y_test) == 2


def test_reproductible() -> None:
    """Deux séparations successives donnent exactement les mêmes lignes.
    Ne reçoit rien. Renvoie : rien, échoue si les résultats diffèrent.
    """
    premier = separer_donnees(creer_petit_jeu())
    deuxieme = separer_donnees(creer_petit_jeu())
    assert list(premier[1].index) == list(deuxieme[1].index)


def test_cible_absente() -> None:
    """Sans la colonne charges, la séparation lève DonneeInvalideError.
    Ne reçoit rien. Renvoie : rien, échoue si l'exception n'est pas levée.
    """
    donnees = creer_petit_jeu().drop(columns=["charges"])
    with pytest.raises(DonneeInvalideError):
        separer_donnees(donnees)