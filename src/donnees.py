import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime
from src.exceptions import DonneeInvalideError
from abc import ABC, abstractmethod

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

def nettoyer_donnees(donnees: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes en double.
    Reçoit : le DataFrame lu par charger_donnees.
    Renvoie : un nouveau DataFrame sans doublons, avec un index renuméroté.
    """
    return donnees.drop_duplicates().reset_index(drop=True)

def graphique_histogramme_frais(donnees: pd.DataFrame, chemin: str) -> None:
    """Enregistre l'histogramme des frais médicaux annuels.
    Reçoit : le DataFrame nettoyé et le chemin du fichier image à créer.
    Renvoie : rien, écrit l'image sur le disque.
    """
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    fig, ax = plt.subplots()
    ax.hist(donnees["charges"], bins=30)
    ax.set_title("Répartition des frais médicaux annuels")
    ax.set_xlabel("Frais annuels ($)")
    ax.set_ylabel("Nombre de personnes")
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def graphique_nuage_imc_frais(donnees: pd.DataFrame, chemin: str) -> None:
    """Enregistre le nuage de points IMC contre frais, coloré selon fumeur.
    Reçoit : le DataFrame nettoyé et le chemin du fichier image à créer.
    Renvoie : rien, écrit l'image sur le disque.
    """
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    fig, ax = plt.subplots()
    for statut, groupe in donnees.groupby("smoker"):
        ax.scatter(groupe["bmi"], groupe["charges"], s=12, alpha=0.6,
                   label=f"Fumeur : {statut}")
    ax.set_title("Frais médicaux selon l'IMC")
    ax.set_xlabel("IMC")
    ax.set_ylabel("Frais annuels ($)")
    ax.legend()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)

ORDRE_VARIABLES = ["age", "sex", "bmi", "children", "smoker", "region"]
LIBELLES = {"age": "Âge (années)", "sex": "Sexe", "bmi": "IMC",
            "children": "Nombre d'enfants", "smoker": "Fumeur", "region": "Région"}


class Variable(ABC):
    """Variable d'entrée du modèle : classe parente abstraite.
    Chaque classe enfant sait valider une saisie à sa façon.
    """

    def __init__(self, nom: str, libelle: str) -> None:
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le jeu de données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché à l'utilisateur."""
        return self._libelle

    @abstractmethod
    def valider(self, saisie: str):
        """Vérifie une saisie et la convertit.
        Reçoit : le texte saisi par l'utilisateur.
        Renvoie : la valeur convertie.
        Lève DonneeInvalideError si la saisie est refusée.
        """

    def __str__(self) -> str:
        return f"{self._libelle} ({self._nom})"


class VariableNumerique(Variable):
    """Variable numérique, acceptée seulement dans un intervalle."""

    def __init__(self, nom: str, libelle: str, minimum: float, maximum: float) -> None:
        super().__init__(nom, libelle)
        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> float:
        """Plus petite valeur vue dans les données."""
        return self._minimum

    @property
    def maximum(self) -> float:
        """Plus grande valeur vue dans les données."""
        return self._maximum

    def valider(self, saisie: str) -> float:
        """Vérifie que la saisie est un nombre dans l'intervalle.
        Reçoit : le texte saisi (la virgule décimale est acceptée).
        Renvoie : le nombre.
        Lève DonneeInvalideError si la saisie est vide, n'est pas un nombre
        ou sort de l'intervalle.
        """
        texte = str(saisie).strip().replace(",", ".")
        if texte == "":
            raise DonneeInvalideError(f"{self._libelle} : la saisie est vide.")
        try:
            valeur = float(texte)
        except ValueError as erreur:
            raise DonneeInvalideError(
                f"{self._libelle} : « {texte} » n'est pas un nombre.") from erreur
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                f"{self._libelle} doit être entre {self._minimum:g} et {self._maximum:g}.")
        return valeur


class VariableCategorielle(Variable):
    """Variable catégorielle, acceptée seulement parmi une liste de valeurs."""

    def __init__(self, nom: str, libelle: str, valeurs: list) -> None:
        super().__init__(nom, libelle)
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list:
        """Valeurs possibles, telles que vues dans les données."""
        return list(self._valeurs)

    def valider(self, saisie: str) -> str:
        """Vérifie que la saisie fait partie des valeurs possibles.
        Reçoit : le texte choisi.
        Renvoie : la valeur choisie.
        Lève DonneeInvalideError si la valeur est vide ou inconnue.
        """
        texte = str(saisie).strip()
        if texte not in self._valeurs:
            raise DonneeInvalideError(
                f"{self._libelle} : choisissez parmi {', '.join(self._valeurs)}.")
        return texte


def creer_variables(metadonnees: dict) -> list:
    """Construit la liste des variables d'entrée à partir des métadonnées.
    Reçoit : le dictionnaire lu dans metadonnees.json.
    Renvoie : une liste de Variable, dans l'ordre d'affichage.
    """
    variables = []
    for nom in ORDRE_VARIABLES:
        if nom in metadonnees["intervalles"]:
            minimum, maximum = metadonnees["intervalles"][nom]
            variables.append(VariableNumerique(nom, LIBELLES[nom], minimum, maximum))
        else:
            variables.append(VariableCategorielle(
                nom, LIBELLES[nom], metadonnees["categories"][nom]))
    return variables

CHEMIN_HISTORIQUE = os.path.join("historique", "predictions.csv")
COLONNES_HISTORIQUE = ["date"] + ORDRE_VARIABLES + ["prediction"]


def ajouter_prediction(chemin: str, saisies: dict, prediction: float) -> None:
    """Ajoute une ligne à l'historique des prédictions (CSV).
    Reçoit : le chemin du fichier, les saisies validées {nom: valeur} et la
    prédiction en dollars. Crée le dossier et l'en-tête si le fichier est neuf.
    Ne renvoie rien.
    Lève DonneeInvalideError si une saisie manque ou si le fichier est illisible.
    """
    try:
        ligne = ([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                 + [saisies[nom] for nom in ORDRE_VARIABLES]
                 + [round(prediction, 2)])
        dossier = os.path.dirname(chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        nouveau = not os.path.exists(chemin)
        with open(chemin, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.writer(fichier)
            if nouveau:
                ecrivain.writerow(COLONNES_HISTORIQUE)
            ecrivain.writerow(ligne)
    except KeyError as erreur:
        raise DonneeInvalideError(f"Saisie manquante : {erreur}") from erreur
    except OSError as erreur:
        raise DonneeInvalideError("Impossible d'écrire l'historique.") from erreur


def lire_historique(chemin: str) -> pd.DataFrame:
    """Relit l'historique des prédictions.
    Reçoit : le chemin du fichier CSV.
    Renvoie : un DataFrame, vide (mais avec les bonnes colonnes) si le fichier
    n'existe pas encore.
    Lève DonneeInvalideError si le fichier est abîmé ou n'a pas les bonnes colonnes.
    """
    if not os.path.exists(chemin):
        return pd.DataFrame(columns=COLONNES_HISTORIQUE)
    try:
        with open(chemin, encoding="utf-8") as fichier:
            historique = pd.read_csv(fichier)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as erreur:
        raise DonneeInvalideError("Le fichier d'historique est abîmé.") from erreur
    if list(historique.columns) != COLONNES_HISTORIQUE:
        raise DonneeInvalideError("Le fichier d'historique n'a pas les bonnes colonnes.")
    return historique

if __name__ == "__main__":
    jeu = nettoyer_donnees(charger_donnees(os.path.join("donnees", "insurance.csv")))
    print(jeu.shape)
    graphique_histogramme_frais(jeu, os.path.join("donnees", "exploration_histogramme.png"))
    graphique_nuage_imc_frais(jeu, os.path.join("donnees", "exploration_nuage.png"))