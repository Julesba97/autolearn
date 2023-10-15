import pandas as  pd
import numpy as np
from pathlib import Path
from ensure import ensure_annotations
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import (FunctionTransformer, 
                                   OneHotEncoder, 
                                  StandardScaler)

from .logger import logger

import warnings
warnings.filterwarnings("ignore")

def replace_categorical_with_frequencies(data):
    result = data.copy() 
    for column in data.columns:
        if data[column].dtype == 'object':
            category_frequencies = data[column].value_counts(normalize=True)
            result[column] = data[column].map(category_frequencies)
    return result

CategoricalFrequencyTransformer = FunctionTransformer(replace_categorical_with_frequencies)

class DataTransformer:
    def __init__(self, alpha=0.05, threshold=5):
        self.alpha=alpha
        self.threshold = threshold
        
    
    @ensure_annotations
    def get_non_normal_columns(self, data:pd.DataFrame)->set:
        """
        Identifie les colonnes non-normalement distribuées dans un DataFrame.

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.

        Returns:
            set: Un ensemble de noms de colonnes non-normalement distribuées.

        Example:
            # Identifie les colonnes non-normalement distribuées
            non_normal_columns = transformer.get_non_normal_columns(data)

        Logs:
            - Pour chaque colonne non-normalement distribuée, un message est affiché.
            Exemple : "La colonne 'feature_name' n'est pas normalement distribuée."

        Note:
            Cette fonction utilise un test statistique de Kolmogorov-Smirnov pour détecter les distributions non-normales.
        """
        data_ = data.copy()
        try:
            numerical_features = data_.select_dtypes(np.number).columns
            if len(numerical_features) !=0 :
                logger.info("Début du test statistique de Kolmogorov-Smirnov pour détecter les distributions non-normales.")
                non_normal_columns = []
                for column in numerical_features:
                    stat, p_value = stats.kstest(data_[column], 'norm')
                    if p_value < self.alpha:
                        non_normal_columns.append(column)
                        logger.info(f"La colonne '{column}' n'est pas normalement distribuée.")

                return set(non_normal_columns)
            else:
                logger.info("Aucune colonne numérique pour effectuer le test de Kolmogorov-Smirnov.")
                return set()
        except Exception as e:
            logger.exception(e)
    
    @ensure_annotations
    def get_limited_category_columns(self, data:pd.DataFrame)->set:
        """
        Identifie les colonnes catégorielles avec un nombre limité défini par 'threshold' de catégories uniques.
        Ces colonnes seront utilisées pour effectuer la transformation en encodage one-hot (OneHotEncoding).
        
        Cette fonction peut être utile pour éviter l'utilisation de toutes les colonnes catégorielles
        avec un grand nombre de modalités, sur lesquelles une transformation en one-hot encoding pourrait
        entraîner une explosion du nombre de variables pour la modélisation.
        Pour ces colonnes, la transformation qui va être utilisée est 'CategoricalFrequencyTransformer', 
        c'est-à-dire remplacer les modalités par leurs fréquences.
        
        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.

        Returns:
            set: Un ensemble de noms de colonnes catégorielles avec un nombre limité de catégories uniques
                inférieur à un seuil défini par 'threshold'.
        Logs:
            - Pour chaque colonne identifiée, un message est affiché.
            Exemple : "La colonne 'feature_name' a un total de {len(unique_values)} modalités."
        """
        data_ = data.copy()
        try:
            categorical_features = data_.select_dtypes(include=['object', 'category']).columns
            if len(categorical_features) != 0:
                low_cardinality_cols = []
                for column in categorical_features:
                    if data_[column].nunique() < self.threshold:
                        low_cardinality_cols.append(column)
                        logger.info(f"La colonne '{column}' a un total de {len(data_[column].unique())} modalités.")
                logger.info(f"Nombre de colonnes catégorielles avec un\
                            nombre de modalités inférieur à {self.threshold}: {len(low_cardinality_cols)}.")

                return set(low_cardinality_cols)
            else:
                logger.info("Aucune colonne catégorielle n'a été détectée")
                return set()
        except Exception as e:
            logger.exception(e)
    
    
    def fit_transform(self, data:pd.DataFrame)->np.ndarray:
        """
        Transforme les données en appliquant différentes méthodes : 
        réalise une transformation logarithmique sur les données qui ne suivent pas une distribution normale,
        effectue un encodage one-hot sur certaines données catégorielles,
        remplace les modalités par leurs fréquences pour d'autres, 
        et normalise les données pour qu'elles soient comprises dans l'intervalle [0, 1].

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données à transformer.

        Returns:
            np.ndarray: Les données transformées sous forme de tableau NumPy.
        """
        data_ = data.copy()
        non_normal_columns = self.get_non_normal_columns(data_)
        logger.info(f"Nombre de variables à transformer avec log transformation et StandardScaler :\
                    {len(non_normal_columns)}.")

        kolmogorov_normal_columns = list(set(data_.select_dtypes(np.number)) - \
                                        non_normal_columns)
        logger.info(f"Nombre de variables à transformer avec StandardScaler :\
                {len(kolmogorov_normal_columns)}.")

        
        limited_modalities_categorical_columns = self.get_limited_category_columns(data_)
        logger.info(f"Nombre de variables à encoder en one-hot : {len(limited_modalities_categorical_columns)}.")
        
        high_cardinality_categorical_columns = list(set(data_.select_dtypes(include=["object", "category"])) - \
                                                   limited_modalities_categorical_columns)
        logger.info(f"Nombre de variables à transformer avec CategoricalFrequencyTransformer :\
                {len(high_cardinality_categorical_columns)}.")
       
        non_normal_columns_transformer = make_pipeline(FunctionTransformer(np.log1p), 
                                                      StandardScaler())
        try:
            if len(non_normal_columns) !=0 and len(limited_modalities_categorical_columns) != 0 and\
            len(kolmogorov_normal_columns)  != 0 and len(high_cardinality_categorical_columns) !=0:
                self.transformation_pipeline = make_column_transformer((StandardScaler(),
                                                        kolmogorov_normal_columns),
                                                                (non_normal_columns_transformer, list(non_normal_columns)),
                                                                (CategoricalFrequencyTransformer,
                                                                    high_cardinality_categorical_columns),
                                                                (OneHotEncoder(handle_unknown="ignore"),
                                                                    list(limited_modalities_categorical_columns)),
                                                                remainder="passthrough")
                
            elif len(non_normal_columns) == 0 and len(limited_modalities_categorical_columns) != 0 and\
            len(high_cardinality_categorical_columns) != 0:
                self.transformation_pipeline = make_column_transformer((StandardScaler(),
                                                        kolmogorov_normal_columns),
                                                                (CategoricalFrequencyTransformer,
                                                                    high_cardinality_categorical_columns),
                                                                (OneHotEncoder(handle_unknown="ignore"),
                                                                    list(limited_modalities_categorical_columns)),
                                                                remainder="passthrough")
            
            elif len(non_normal_columns) == 0 and len(limited_modalities_categorical_columns) == 0:
                self.transformation_pipeline = make_column_transformer((StandardScaler(),
                                                        kolmogorov_normal_columns),
                                                                (CategoricalFrequencyTransformer,
                                                                    high_cardinality_categorical_columns),
                                                                remainder="passthrough")
                
            elif len(non_normal_columns) == 0 and len(high_cardinality_categorical_columns) == 0:
                self.transformation_pipeline = make_column_transformer((StandardScaler(),
                                                        kolmogorov_normal_columns),
                                                                (OneHotEncoder(handle_unknown="ignore"),
                                                                    list(limited_modalities_categorical_columns)),
                                                                remainder="passthrough")
            
            elif len(kolmogorov_normal_columns) == 0 and len(limited_modalities_categorical_columns) != 0 and\
            len(high_cardinality_categorical_columns) != 0:
                self.transformation_pipeline = make_column_transformer(
                                                                (non_normal_columns_transformer, list(non_normal_columns)),
                                                                (CategoricalFrequencyTransformer,
                                                                    high_cardinality_categorical_columns),
                                                                (OneHotEncoder(handle_unknown="ignore"),
                                                                    list(limited_modalities_categorical_columns)),
                                                                remainder="passthrough")
                    
                    
            elif len(kolmogorov_normal_columns) == 0 and len(limited_modalities_categorical_columns) == 0:
                self.transformation_pipeline = make_column_transformer(
                                                                (non_normal_columns_transformer, list(non_normal_columns)),
                                                                (CategoricalFrequencyTransformer,
                                                                    high_cardinality_categorical_columns),
                                                                remainder="passthrough")  
            
            elif len(kolmogorov_normal_columns) == 0 and len(high_cardinality_categorical_columns) == 0:
                self.transformation_pipeline = make_column_transformer(
                                                                (non_normal_columns_transformer, list(non_normal_columns)),
                                                                (OneHotEncoder(handle_unknown="ignore"),
                                                                    list(limited_modalities_categorical_columns)),
                                                                remainder="passthrough")
            
            elif len(non_normal_columns) !=0 and len(limited_modalities_categorical_columns) == 0 and\
            len(kolmogorov_normal_columns)  != 0 :
                self.transformation_pipeline = make_column_transformer((StandardScaler(),
                                                        kolmogorov_normal_columns),
                                                                (non_normal_columns_transformer, list(non_normal_columns)),
                                                                (CategoricalFrequencyTransformer,
                                                                    high_cardinality_categorical_columns),
                                                                remainder="passthrough")
            
            
            elif len(non_normal_columns) !=0  and len(kolmogorov_normal_columns)  != 0 and\
            len(high_cardinality_categorical_columns) ==0 and len(limited_modalities_categorical_columns) !=0:
                self.transformation_pipeline = make_column_transformer((StandardScaler(),
                                                        kolmogorov_normal_columns),
                                                                (non_normal_columns_transformer, list(non_normal_columns)),
                                                                (OneHotEncoder(handle_unknown="ignore"),
                                                                    list(limited_modalities_categorical_columns)),
                                                                remainder="passthrough")
        
            self.transformed_data = self.transformation_pipeline.fit_transform(data_)
            
            return self.transformed_data
        except Exception as e:
            logger.exception(e)
    
    @ensure_annotations    
    def transform(self, test_data:pd.DataFrame)->np.ndarray:
        """
        Applique les mêmes transformations sur l'ensemble de test que celles appliquées
        sur l'ensemble d'entraînement.
        
        Cette fonction applique les mêmes transformations (encodage, normalisation, etc.)
        sur l'ensemble de test que celles qui ont été appliquées sur l'ensemble d'entraînement.


        Parameters:
            test_data (pd.DataFrame): Le DataFrame contenant les données de l'ensemble de test.

        Returns:
            np.ndarray: Les données transformées de l'ensemble de test sous forme de tableau NumPy.
            
        """
        try:
            logger.info("Les données de test transformées ont été retournées.")
            return self.transformation_pipeline.transform(test_data)
        except Exception as e:
            logger.exception(e)