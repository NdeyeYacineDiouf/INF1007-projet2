import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import DonneeInvalideError

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
