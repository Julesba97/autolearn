import pandas as pd
from pathlib import Path

from .logger import logger

def load_data(filepath:Path) ->pd.DataFrame:
    """
    Charge les données à partir d'un fichier et les renvoie sous forme de DataFrame.

    Parameters:
        filepath (Path): Le chemin du fichier à charger.

    Returns:
        pd.DataFrame: Le DataFrame contenant les données.

    Example:
        # Charge les données à partir du fichier 'data.csv'
        data = load_data(Path('data.csv'))

    Note:
        Le fichier doit être au format pris en charge par pandas CSV.
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.exception(e)