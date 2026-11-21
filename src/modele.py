import pandas as pd
import numpy as np
import functools
import time
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from src.exceptions import DonneeInvalideError
from sklearn.linear_model import LinearRegression

COLONNE_CIBLE = "charges"
COLONNES_CATEGORIELLES = ["sex", "smoker", "region"]
GRAINE = 42
PROPORTION_TEST = 0.2
COLONNES_NUMERIQUES = ["age", "bmi", "children"]
NOM_MODELE = "modele.joblib"
NOM_METADONNEES = "metadonnees.json"

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

def chronometrer(fonction):
    """Décorateur : affiche le temps d'exécution de la fonction décorée.
    Reçoit : la fonction à chronométrer.
    Renvoie : une fonction qui fait la même chose et affiche sa durée.
    """
    @functools.wraps(fonction)
    def enveloppe(*args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        print(f"{fonction.__name__} : {time.perf_counter() - debut:.3f} s")
        return resultat
    return enveloppe

@chronometrer
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

def construire_metadonnees(donnees: pd.DataFrame, colonnes: list, metriques: dict) -> dict:
    """Décrit le modèle pour que l'interface puisse se construire.
    Reçoit : le DataFrame nettoyé, les noms des colonnes d'entrée encodées
    (dans l'ordre vu par le modèle) et les métriques obtenues.
    Renvoie : un dictionnaire avec les colonnes, les valeurs possibles de
    chaque variable catégorielle, l'intervalle [min, max] de chaque variable
    numérique et les métriques.
    """
    return {
        "colonnes": list(colonnes),
        "categories": {c: sorted(donnees[c].unique().tolist())
                       for c in COLONNES_CATEGORIELLES},
        "intervalles": {c: [float(donnees[c].min()), float(donnees[c].max())]
                        for c in COLONNES_NUMERIQUES},
        "metriques": metriques,
    }            

def sauvegarder_modele(modele: LinearRegression, metadonnees: dict, dossier: str) -> None:
    """Enregistre le modèle (joblib) et ses métadonnées (JSON) dans un dossier.
    Reçoit : le modèle entraîné, le dictionnaire de métadonnées et le dossier.
    Ne renvoie rien : crée le dossier au besoin et écrit les deux fichiers.
    """
    os.makedirs(dossier, exist_ok=True)
    joblib.dump(modele, os.path.join(dossier, NOM_MODELE))
    with open(os.path.join(dossier, NOM_METADONNEES), "w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)

def charger_modele(dossier: str) -> tuple:
    """Recharge le modèle et ses métadonnées sans ré-entraîner.
    Reçoit : le dossier où sauvegarder_modele a écrit les fichiers.
    Renvoie : le tuple (modele, metadonnees).
    Lève DonneeInvalideError si un fichier est absent ou si le JSON est abîmé.
    """
    try:
        modele = joblib.load(os.path.join(dossier, NOM_MODELE))
        with open(os.path.join(dossier, NOM_METADONNEES), encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)
    except FileNotFoundError as erreur:
        raise DonneeInvalideError(
            "Modèle introuvable : lancez d'abord python -m src.modele.") from erreur
    except json.JSONDecodeError as erreur:
        raise DonneeInvalideError("Le fichier de métadonnées est abîmé.") from erreur
    return modele, metadonnees

class Modele:
    """Modèle entraîné et ses métadonnées, prêt à prédire.
    L'interface s'en sert sans connaître scikit-learn.
    """

    def __init__(self, modele: LinearRegression, metadonnees: dict) -> None:
        self._modele = modele
        self._metadonnees = metadonnees

    @classmethod
    def depuis_dossier(cls, dossier: str) -> "Modele":
        """Charge le modèle sauvegardé et ses métadonnées.
        Reçoit : le dossier écrit par sauvegarder_modele.
        Renvoie : un objet Modele.
        Lève DonneeInvalideError si les fichiers sont absents ou abîmés.
        """
        modele, metadonnees = charger_modele(dossier)
        return cls(modele, metadonnees)

    @property
    def metadonnees(self) -> dict:
        """Métadonnées du modèle (colonnes, catégories, intervalles, métriques)."""
        return self._metadonnees

    @property
    def rmse(self) -> float:
        """Erreur typique du modèle sur le test, en dollars."""
        try:
            return float(self._metadonnees["metriques"]["regression"]["rmse"])
        except KeyError as erreur:
            raise DonneeInvalideError("Le RMSE est absent des métadonnées.") from erreur

    def predire(self, saisies: dict) -> float:
        """Prédit les frais annuels à partir de saisies déjà validées.
        Reçoit : un dictionnaire {nom de variable: valeur}, par exemple
        {"age": 35.0, "sex": "male", "smoker": "yes", ...}.
        Renvoie : les frais prédits en dollars, jamais négatifs.
        Lève DonneeInvalideError si une variable est absente des saisies.
        """
        valeurs = []
        colonnes = self._metadonnees["colonnes"]
        try:
            for colonne in colonnes:
                if colonne in saisies:
                    valeurs.append(float(saisies[colonne]))
                elif "_" in colonne:
                    nom, categorie = colonne.split("_", 1)
                    valeurs.append(1.0 if saisies[nom] == categorie else 0.0)
                else:
                    raise DonneeInvalideError(f"Saisie manquante : {colonne}")
        except KeyError as erreur:
            raise DonneeInvalideError(f"Saisie manquante : {erreur}") from erreur
        entrees = pd.DataFrame([valeurs], columns=colonnes)
        return max(0.0, float(self._modele.predict(entrees)[0]))

    def fourchette(self, prediction: float) -> tuple:
        """Donne une fourchette autour d'une prédiction : ± le RMSE.
        Reçoit : une prédiction en dollars.
        Renvoie : le tuple (bas, haut), le bas ne descendant pas sous 0.
        """
        return max(0.0, prediction - self.rmse), prediction + self.rmse

    def __str__(self) -> str:
        return (f"Modèle linéaire ({len(self._metadonnees['colonnes'])} colonnes, "
                f"RMSE {self.rmse:.0f} $)")


if __name__ == "__main__":
    from src.donnees import charger_donnees, nettoyer_donnees

    jeu = nettoyer_donnees(charger_donnees(os.path.join("donnees", "insurance.csv")))
    x_entr, x_test, y_entr, y_test = separer_donnees(jeu)
    reference = evaluer_reference(y_entr, y_test)
    print("Référence (moyenne) :", reference)
    modele = entrainer_modele(x_entr, y_entr)
    resultat = evaluer_modele(modele, x_test, y_test)
    print("Régression linéaire :", resultat)
    metadonnees = construire_metadonnees(
        jeu, list(x_entr.columns), {"reference": reference, "regression": resultat})
    sauvegarder_modele(modele, metadonnees, "modeles")
    print("Modèle sauvegardé dans modeles/")
    print("Régression linéaire :", evaluer_modele(modele, x_test, y_test))