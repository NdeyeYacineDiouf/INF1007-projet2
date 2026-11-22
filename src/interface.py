import os
import pandas as pd
from matplotlib.figure import Figure
from src.donnees import (
    CHEMIN_HISTORIQUE,
    ajouter_prediction,
    charger_donnees,
    creer_variables,
    graphique_prediction,
    lire_historique,
    nettoyer_donnees,
)
from src.exceptions import DonneeInvalideError
from src.modele import Modele

CHEMIN_DONNEES = os.path.join("donnees", "insurance.csv")
DOSSIER_MODELE = "modeles"


class Application:
    """Logique de l'application : valider les saisies, prédire, garder l'historique.
    Elle contient un Modele, la liste des variables et les frais observés.
    Aucun code d'affichage ici : la page Streamlit n'a qu'à l'appeler.
    """

    def __init__(self, modele: Modele, variables: list, frais: pd.Series,
                 chemin_historique: str = CHEMIN_HISTORIQUE) -> None:
        self._modele = modele
        self._variables = variables
        self._frais = frais
        self._chemin_historique = chemin_historique

    @classmethod
    def depuis_fichiers(cls, dossier_modele: str = DOSSIER_MODELE,
                        chemin_donnees: str = CHEMIN_DONNEES,
                        chemin_historique: str = CHEMIN_HISTORIQUE) -> "Application":
        """Construit l'application à partir des fichiers sur le disque.
        Reçoit : le dossier du modèle sauvegardé, le CSV et le chemin de l'historique.
        Renvoie : une Application prête à l'emploi.
        Lève DonneeInvalideError si le modèle ou le jeu de données est introuvable.
        """
        modele = Modele.depuis_dossier(dossier_modele)
        try:
            frais = nettoyer_donnees(charger_donnees(chemin_donnees))["charges"]
        except FileNotFoundError as erreur:
            raise DonneeInvalideError(
                f"Jeu de données introuvable : {chemin_donnees}") from erreur
        variables = creer_variables(modele.metadonnees)
        return cls(modele, variables, frais, chemin_historique)

    @property
    def variables(self) -> list:
        """Variables d'entrée, dans l'ordre d'affichage."""
        return list(self._variables)

    @property
    def modele(self) -> Modele:
        """Le modèle utilisé pour prédire."""
        return self._modele

    def valider_saisies(self, saisies: dict) -> dict:
        """Valide toutes les saisies en une fois.
        Reçoit : un dictionnaire {nom de variable: texte saisi}. Une variable
        absente du dictionnaire compte comme une saisie vide.
        Renvoie : un dictionnaire {nom: valeur convertie}.
        Lève DonneeInvalideError avec un message par saisie refusée, une par ligne.
        """
        valeurs = {}
        erreurs = []
        for variable in self._variables:
            try:
                valeurs[variable.nom] = variable.valider(saisies.get(variable.nom, ""))
            except DonneeInvalideError as erreur:
                erreurs.append(str(erreur))
        if erreurs:
            raise DonneeInvalideError("\n".join(erreurs))
        return valeurs

    def predire(self, saisies: dict) -> dict:
        """Valide les saisies, prédit les frais et ajoute la ligne à l'historique.
        Reçoit : un dictionnaire {nom de variable: texte saisi}.
        Renvoie : {"prediction": frais en dollars, "fourchette": (bas, haut)}.
        Lève DonneeInvalideError si une saisie est refusée.
        """
        valeurs = self.valider_saisies(saisies)
        prediction = self._modele.predire(valeurs)
        ajouter_prediction(self._chemin_historique, valeurs, prediction)
        return {"prediction": prediction,
                "fourchette": self._modele.fourchette(prediction)}

    def graphique(self, prediction: float, fourchette: tuple) -> Figure:
        """Construit le graphique de la prédiction parmi les frais observés.
        Reçoit : la prédiction et sa fourchette.
        Renvoie : la figure Matplotlib.
        """
        return graphique_prediction(self._frais, prediction, fourchette)

    def historique(self) -> pd.DataFrame:
        """Relit l'historique des prédictions.
        Renvoie : un DataFrame, vide s'il n'y a encore aucune prédiction.
        """
        return lire_historique(self._chemin_historique)

    def __str__(self) -> str:
        return f"Application ({len(self._variables)} variables) — {self._modele}"