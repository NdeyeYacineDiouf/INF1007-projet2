from pathlib import Path
import pytest

from src.donnees import charger_donnees
from src.exceptions import DonneeInvalideError

CSV_VALIDE = (
    "age,sex,bmi,children,smoker,region,charges\n"
    "19,female,27.9,0,yes,southwest,16884.92\n"
    "33,male,22.7,0,no,northwest,21984.47\n"
)


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