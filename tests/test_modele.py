import pandas as pd
import pytest
import numpy as np
from src.exceptions import DonneeInvalideError
from src.modele import (calculer_r2, calculer_rmse, encoder_categories, evaluer_reference, separer_donnees, evaluer_modele, entrainer_modele)
from sklearn.metrics import mean_squared_error, r2_score

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

def test_rmse() -> None:
    """Le RMSE de [1, 2, 3] contre [1, 2, 5] vaut racine de 4/3.
    Ne reçoit rien. Renvoie : rien, échoue si la valeur est fausse.
    """
    assert calculer_rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_metriques_comme_sklearn() -> None:
    """Nos métriques NumPy coïncident avec celles de scikit-learn.
    Ne reçoit rien. Renvoie : rien, échoue si les valeurs diffèrent.
    """
    reel = np.array([3.0, 5.0, 2.0, 8.0])
    predit = np.array([2.5, 5.5, 2.0, 7.0])
    assert calculer_rmse(reel, predit) == pytest.approx(np.sqrt(mean_squared_error(reel, predit)))
    assert calculer_r2(reel, predit) == pytest.approx(r2_score(reel, predit))


def test_reference_moyenne() -> None:
    """La référence prédit la moyenne de l'entraînement (2) : RMSE 1 et R² 0.
    Ne reçoit rien. Renvoie : rien, échoue si une valeur est fausse.
    """
    resultat = evaluer_reference([1.0, 2.0, 3.0], [1.0, 3.0])
    assert resultat["rmse"] == pytest.approx(1.0)
    assert resultat["r2"] == pytest.approx(0.0)


def test_r2_valeurs_identiques() -> None:
    """Un R² sur des valeurs réelles toutes égales lève DonneeInvalideError.
    Ne reçoit rien. Renvoie : rien, échoue si l'exception n'est pas levée.
    """
    with pytest.raises(DonneeInvalideError):
        calculer_r2([2.0, 2.0], [2.0, 3.0])

def test_regression_parfaite() -> None:
    """Sur des données parfaitement linéaires (y = 2x), le R² vaut 1.
    Ne reçoit rien. Renvoie : rien, échoue si le R² est différent de 1.
    """
    entrees = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    cible = pd.Series([2.0, 4.0, 6.0, 8.0])
    modele = entrainer_modele(entrees, cible)
    assert modele.coef_[0] == pytest.approx(2.0)
    assert evaluer_modele(modele, entrees, cible)["r2"] == pytest.approx(1.0)

def test_modele_predit_nouvelles_valeurs() -> None:
    """Un modèle entraîné sur y = 2x prédit 10 pour x = 5.
    Ne reçoit rien. Renvoie : rien, échoue si la prédiction est fausse.
    """
    entrees = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    modele = entrainer_modele(entrees, pd.Series([2.0, 4.0, 6.0, 8.0]))
    assert modele.predict(pd.DataFrame({"x": [5.0]}))[0] == pytest.approx(10.0)