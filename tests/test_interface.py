from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.donnees import creer_variables
from src.exceptions import DonneeInvalideError
from src.interface import Application
from src.modele import Modele, entrainer_modele

METADONNEES = {
    "intervalles": {"age": [18.0, 64.0], "bmi": [16.0, 53.0], "children": [0.0, 5.0]},
    "categories": {"sex": ["female", "male"], "smoker": ["no", "yes"],
                   "region": ["northeast", "northwest"]},
    "colonnes": ["age", "smoker_yes"],
    "metriques": {"regression": {"rmse": 50.0, "r2": 1.0}},
}

SAISIES = {"age": "35", "sex": "male", "bmi": "28,5", "children": "2",
           "smoker": "yes", "region": "northwest"}


def creer_application(chemin_historique: str) -> Application:
    """Crée une Application avec un petit modèle : frais = 10 x âge + 500 si fumeur.
    Reçoit : le chemin du fichier d'historique à utiliser.
    Renvoie : une Application prête pour les tests.
    """
    entrees = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0],
                            "smoker_yes": [0.0, 1.0, 0.0, 1.0]})
    cible = pd.Series([200.0, 800.0, 400.0, 1000.0])
    modele = Modele(entrainer_modele(entrees, cible), METADONNEES)
    frais = pd.Series([200.0, 400.0, 800.0, 1000.0])
    return Application(modele, creer_variables(METADONNEES), frais, chemin_historique)


def test_predire_et_enregistrer(tmp_path: Path) -> None:
    """Des saisies valides donnent la prédiction, la fourchette et une ligne d'historique.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si un résultat est faux.
    """
    application = creer_application(str(tmp_path / "h.csv"))
    resultat = application.predire(SAISIES)
    assert resultat["prediction"] == pytest.approx(850.0)
    assert resultat["fourchette"] == (pytest.approx(800.0), pytest.approx(900.0))
    assert len(application.historique()) == 1


def test_saisies_invalides_toutes_signalees(tmp_path: Path) -> None:
    """Deux saisies refusées donnent un seul message qui cite les deux, sans historique.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si un message manque ou si un fichier est créé.
    """
    chemin = tmp_path / "h.csv"
    application = creer_application(str(chemin))
    with pytest.raises(DonneeInvalideError) as erreur:
        application.predire({**SAISIES, "age": "", "bmi": "abc"})
    assert "Âge" in str(erreur.value)
    assert "IMC" in str(erreur.value)
    assert not chemin.exists()


def test_saisies_absentes(tmp_path: Path) -> None:
    """Un dictionnaire vide est refusé comme un ensemble de saisies vides.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si l'exception n'est pas levée.
    """
    with pytest.raises(DonneeInvalideError):
        creer_application(str(tmp_path / "h.csv")).predire({})


def test_historique_vide_au_depart(tmp_path: Path) -> None:
    """Sans prédiction, l'historique est vide.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si le tableau n'est pas vide.
    """
    assert len(creer_application(str(tmp_path / "h.csv")).historique()) == 0


def test_graphique(tmp_path: Path) -> None:
    """Le graphique de l'application est une figure avec un titre.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si le titre est vide.
    """
    fig = creer_application(str(tmp_path / "h.csv")).graphique(850.0, (800.0, 900.0))
    assert fig.axes[0].get_title() != ""
    plt.close(fig)


def test_depuis_fichiers_modele_absent(tmp_path: Path) -> None:
    """Un dossier sans modèle lève DonneeInvalideError.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si l'exception n'est pas levée.
    """
    with pytest.raises(DonneeInvalideError):
        Application.depuis_fichiers(dossier_modele=str(tmp_path))