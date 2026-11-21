from pathlib import Path
import pytest
import pandas as pd

from src.donnees import (Variable,VariableCategorielle,VariableNumerique,charger_donnees,creer_variables,nettoyer_donnees)
from src.exceptions import DonneeInvalideError

CSV_VALIDE = (
    "age,sex,bmi,children,smoker,region,charges\n"
    "19,female,27.9,0,yes,southwest,16884.92\n"
    "33,male,22.7,0,no,northwest,21984.47\n"
)

METADONNEES = {
    "intervalles": {"age": [18.0, 64.0], "bmi": [16.0, 53.0], "children": [0.0, 5.0]},
    "categories": {"sex": ["female", "male"], "smoker": ["no", "yes"],
                   "region": ["northeast", "northwest"]},
}

def test_usuel(tmp_path: Path) -> None:
    """Un CSV valide est lu avec 2 lignes et 7 colonnes.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si une vérification est fausse.
    """
    fichier = tmp_path / "ok.csv"
    fichier.write_text(CSV_VALIDE, encoding="utf-8")
    donnees = charger_donnees(str(fichier))
    assert donnees.shape == (2, 7)
    assert donnees["age"].iloc[0] == 19


def test_absent(tmp_path: Path) -> None:
    """Un fichier qui n'existe pas lève FileNotFoundError.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    renvoie : rien, échoue si l'exception n'est pas levée.
    """
    with pytest.raises(FileNotFoundError):
        charger_donnees(str(tmp_path / "absent.csv"))


def test_colonnes(tmp_path: Path) -> None:
    """Un CSV auquel il manque des colonnes lève DonneeInvalideError.
    Reçoit : tmp_path, un dossier temporaire fourni par pytest.
    Renvoie : rien, échoue si l'exception n'est pas levée.
    """
    fichier = tmp_path / "incomplet.csv"
    fichier.write_text("age,sex\n19,female\n", encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        charger_donnees(str(fichier))

def test_doublons() -> None:
    """Une ligne en double est supprimée par nettoyer_donnees.
    Ne reçoit rien. Renvoie : rien, échoue si le nombre de lignes est faux.
    """
    donnees = pd.DataFrame({"age": [19, 19, 33], "charges": [1.0, 1.0, 2.0]})
    assert len(nettoyer_donnees(donnees)) == 2


def test_numerique_valide() -> None:
    """Une saisie valide est convertie, avec espaces ou virgule décimale.
    Ne reçoit rien. Renvoie : rien, échoue si la conversion est fausse.
    """
    age = VariableNumerique("age", "Âge", 18, 64)
    assert age.valider(" 30 ") == pytest.approx(30.0)
    assert age.valider("30,5") == pytest.approx(30.5)


@pytest.mark.parametrize("saisie", ["", "abc", "10", "70", "nan"])
def test_numerique_refuse(saisie: str) -> None:
    """Saisie vide, non numérique ou hors intervalle : DonneeInvalideError.
    Reçoit : une saisie invalide. Renvoie : rien, échoue sans exception.
    """
    with pytest.raises(DonneeInvalideError):
        VariableNumerique("age", "Âge", 18, 64).valider(saisie)


def test_categorielle() -> None:
    """Une valeur connue passe, une valeur inconnue ou vide est refusée.
    Ne reçoit rien. Renvoie : rien, échoue si le comportement est faux.
    """
    fumeur = VariableCategorielle("smoker", "Fumeur", ["no", "yes"])
    assert fumeur.valider("yes") == "yes"
    for saisie in ["maybe", ""]:
        with pytest.raises(DonneeInvalideError):
            fumeur.valider(saisie)


def test_variable_abstraite() -> None:
    """La classe parente Variable ne peut pas être instanciée.
    Ne reçoit rien. Renvoie : rien, échoue si TypeError n'est pas levée.
    """
    with pytest.raises(TypeError):
        Variable("age", "Âge")


def test_creer_variables() -> None:
    """Les variables sont créées dans l'ordre, avec le bon type.
    Ne reçoit rien. Renvoie : rien, échoue si l'ordre ou un type est faux.
    """
    variables = creer_variables(METADONNEES)
    assert [v.nom for v in variables] == ["age", "sex", "bmi", "children", "smoker", "region"]
    assert isinstance(variables[0], VariableNumerique)
    assert isinstance(variables[1], VariableCategorielle)
    assert variables[0].maximum == pytest.approx(64.0)


def test_texte_variable() -> None:
    """__str__ affiche le libellé suivi du nom de la colonne.
    Ne reçoit rien. Renvoie : rien, échoue si le texte est faux.
    """
    assert str(VariableNumerique("bmi", "IMC", 16, 53)) == "IMC (bmi)"