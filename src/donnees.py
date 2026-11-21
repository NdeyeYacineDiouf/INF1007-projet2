import os
import pandas as pd
import matplotlib.pyplot as plt
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

def nettoyer_donnees(donnees: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes en double.
    Reçoit : le DataFrame lu par charger_donnees.
    Renvoie : un nouveau DataFrame sans doublons, avec un index renuméroté.
    """
    return donnees.drop_duplicates().reset_index(drop=True)

def graphique_histogramme_frais(donnees: pd.DataFrame, chemin: str) -> None:
    """Enregistre l'histogramme des frais médicaux annuels.
    Reçoit : le DataFrame nettoyé et le chemin du fichier image à créer.
    Ne renvoie rien : écrit l'image sur le disque.
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
    Ne renvoie rien : écrit l'image sur le disque.
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


if __name__ == "__main__":
    jeu = nettoyer_donnees(charger_donnees(os.path.join("donnees", "insurance.csv")))
    print(jeu.shape)
    graphique_histogramme_frais(jeu, os.path.join("donnees", "exploration_histogramme.png"))
    graphique_nuage_imc_frais(jeu, os.path.join("donnees", "exploration_nuage.png"))