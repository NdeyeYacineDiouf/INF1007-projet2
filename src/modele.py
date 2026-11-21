import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.exceptions import DonneeInvalideError
from sklearn.linear_model import LinearRegression

COLONNE_CIBLE = "charges"
COLONNES_CATEGORIELLES = ["sex", "smoker", "region"]
GRAINE = 42
PROPORTION_TEST = 0.2


def encoder_categories(entrees: pd.DataFrame) -> pd.DataFrame:
    """Remplace chaque colonne catégorielle par des colonnes de 0 et de 1.
    Reçoit : le DataFrame des colonnes d'entrée (sans la cible).
    Renvoie : un nouveau DataFrame ne contenant que des nombres. La première
    catégorie de chaque colonne est omise, car elle se déduit des autres
    (par exemple, sex_male vaut 0 pour une femme).
    """
    return pd.get_dummies(entrees, columns=COLONNES_CATEGORIELLES,
                          drop_first=True, dtype=float)


def separer_donnees(donnees: pd.DataFrame):
    """Sépare les données en un ensemble d'entraînement et un ensemble de test.
    Reçoit : le DataFrame nettoyé, avec la colonne cible.
    Renvoie : quatre éléments (entrees_entrainement, entrees_test,
    cible_entrainement, cible_test), dans cet ordre. Le test contient 20 %
    des lignes, choisies au hasard avec une graine fixée : le résultat est
    donc le même à chaque exécution.
    Lève DonneeInvalideError si la colonne cible est absente.
    """
    if COLONNE_CIBLE not in donnees.columns:
        raise DonneeInvalideError(f"Colonne cible absente : {COLONNE_CIBLE}")

    entrees = encoder_categories(donnees.drop(columns=[COLONNE_CIBLE]))
    cible = donnees[COLONNE_CIBLE]
    return train_test_split(entrees, cible, test_size=PROPORTION_TEST,
                            random_state=GRAINE)

def calculer_rmse(reel: np.ndarray, predit: np.ndarray) -> float:
    """Calcule l'erreur quadratique moyenne (RMSE), en dollars.
    Reçoit : les vraies valeurs et les valeurs prédites, de même longueur.
    Renvoie : la racine de la moyenne des carrés des erreurs.
    """
    return float(np.sqrt(np.mean((np.asarray(reel) - np.asarray(predit)) ** 2)))


def calculer_r2(reel: np.ndarray, predit: np.ndarray) -> float:
    """Calcule le coefficient de détermination R².
    Reçoit : les vraies valeurs et les valeurs prédites, de même longueur.
    Renvoie : 1 pour une prédiction parfaite, 0 pour une prédiction aussi
    mauvaise que la moyenne, négatif si elle est pire.
    Lève DonneeInvalideError si toutes les vraies valeurs sont identiques
    (le R² n'est pas défini dans ce cas).
    """
    reel = np.asarray(reel)
    erreurs = np.sum((reel - np.asarray(predit)) ** 2)
    ecarts = np.sum((reel - np.mean(reel)) ** 2)
    if ecarts == 0:
        raise DonneeInvalideError("R² indéfini : toutes les valeurs réelles sont égales.")
    return float(1 - erreurs / ecarts)


def evaluer_reference(cible_entrainement, cible_test) -> dict:
    """Évalue le modèle trivial qui prédit toujours la moyenne de l'entraînement.
    Reçoit : les frais de l'ensemble d'entraînement et ceux du test.
    Renvoie : un dictionnaire {"rmse": ..., "r2": ...} calculé sur le test.
    """
    predit = np.full(len(cible_test), np.mean(cible_entrainement))
    return {"rmse": calculer_rmse(cible_test, predit),
            "r2": calculer_r2(cible_test, predit)}

def entrainer_modele(entrees_entrainement, cible_entrainement) -> LinearRegression:
    """Entraîne une régression linéaire sur l'ensemble d'entraînement.
    Reçoit : les colonnes d'entrée encodées et les frais correspondants.
    Renvoie : le modèle entraîné.
    """
    modele = LinearRegression()
    modele.fit(entrees_entrainement, cible_entrainement)
    return modele


def evaluer_modele(modele: LinearRegression, entrees_test, cible_test) -> dict:
    """Évalue le modèle entraîné sur l'ensemble de test.
    Reçoit : le modèle, les colonnes d'entrée du test et les frais du test.
    Renvoie : un dictionnaire {"rmse": ..., "r2": ...}.
    """
    predit = modele.predict(entrees_test)
    return {"rmse": calculer_rmse(cible_test, predit),
            "r2": calculer_r2(cible_test, predit)}

if __name__ == "__main__":
    import os
    from src.donnees import charger_donnees, nettoyer_donnees

    jeu = nettoyer_donnees(charger_donnees(os.path.join("donnees", "insurance.csv")))
    x_entr, x_test, y_entr, y_test = separer_donnees(jeu)
    print("Référence (moyenne) :", evaluer_reference(y_entr, y_test))
    modele = entrainer_modele(x_entr, y_entr)
    print("Régression linéaire :", evaluer_modele(modele, x_test, y_test))