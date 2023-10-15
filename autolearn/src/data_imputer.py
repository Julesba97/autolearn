import pandas as  pd
import numpy as np
from pathlib import Path
from ensure import ensure_annotations

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_transformer

from .logger import logger

class DataImputer:
    def __init__(self):
        pass
   
    @ensure_annotations
    def get_missing_categorical_columns(self, data:pd.DataFrame)->dict:
        """
        Identifie les colonnes catégorielles avec des valeurs manquantes dans un DataFrame.

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.

        Returns:
            dict: Un dictionnaire indiquant le nombre de valeurs manquantes par colonne catégorielle.
        """
        self.raw_train_data = data.copy()
        data_ = data.copy()
        try:
            df_categorical = data_.select_dtypes("object")
            if df_categorical.shape[1] !=0:
                logger.info(f"Il y a {df_categorical.shape[1]} colonnes catégorielles dans le jeu de données.")
                categorical_columns_with_missing_values = df_categorical.columns[df_categorical.isna().sum()>0]
                logger.info(f"Il y a {len(categorical_columns_with_missing_values)} colonnes catégorielles avec des valeurs manquantes.")
                missing_values_per_column = df_categorical[categorical_columns_with_missing_values].isna().sum().to_dict()
                if len(missing_values_per_column) !=0:
                    for column, missing_count in missing_values_per_column.items():
                        logger.info(f"La colonne '{column}' a {missing_count} valeurs manquantes.")
                    return missing_values_per_column
                else:
                    logger.info("Aucune colonne catégorielle ne présente de valeurs manquantes.")
                    return dict()
            else:
               logger.info("Aucune colonne catégorielle n'a été trouvée dans le jeu de données.")
               return dict()
        except Exception as e:
            logger.exception(e)
        
    
    @ensure_annotations
    def get_missing_numerical_columns(self, data:pd.DataFrame)->dict:
        """
        Identifie les colonnes numériques avec des valeurs manquantes dans un DataFrame.

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.

        Returns:
            dict: Un dictionnaire indiquant le nombre de valeurs manquantes par colonne numérique.
        """
        self.raw_train_data = data.copy()
        data_ = data.copy()
        try:
            df_numerical = data_.select_dtypes(np.number)
            if df_numerical.shape[1] != 0:
                logger.info(f"Il y a {df_numerical.shape[1]} colonnes numériques dans le jeu de données.")
                numerical_columns_with_missing_values  = df_numerical.columns[df_numerical.isna().sum()>0]
                logger.info(f"Il y a {len(numerical_columns_with_missing_values)} colonnes numériques avec des valeurs manquantes.")
                missing_values_per_column = df_numerical[numerical_columns_with_missing_values].isna().sum().to_dict()
                if len(missing_values_per_column) !=0:
                    for column, missing_count in missing_values_per_column.items():
                        logger.info(f"La colonne '{column}' a {missing_count} valeurs manquantes.")
                    return missing_values_per_column
                else:
                    logger.info("Aucune colonne numérique ne présente de valeurs manquantes.")
                    return dict()
                    
            else:
               logger.info("Aucune colonne numérique n'a été trouvée dans le jeu de données.")
               return dict()
        except Exception as e:
            logger.exception(e)
    
    @ensure_annotations
    def fit_transform_categorical(self, data:pd.DataFrame, return_whole_dataframe=True) -> pd.DataFrame:
        """
        Remplace les valeurs manquantes dans les colonnes catégorielles d'un DataFrame.

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.
            return_whole_dataframe (bool): Indique si le DataFrame complet doit être retourné. Par défaut, True.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes remplacées.
        """
        if not isinstance(data, pd.DataFrame):
            logger.info("'data' doit être un objet de type DataFrame.")
        df_copie = data.copy()
        self._list_categorical_columns_with_missing_values = list(self.get_missing_categorical_columns(df_copie).keys())
        try:
            if len(self._list_categorical_columns_with_missing_values) !=0:
                self.categorical_imputer = make_column_transformer((SimpleImputer(strategy="constant",
                                                                            fill_value="missing"),
                                                            self._list_categorical_columns_with_missing_values), 
                                                            remainder="drop")
                array_with_imputed_values = self.categorical_imputer.fit_transform(df_copie)
                logger.info("Les valeurs manquantes dans les colonnes catégorielles ont été imputées.")
                if return_whole_dataframe:
                    logger.info("Le DataFrame complet avec les valeurs manquantes dans les colonnes catégorielles, remplacées a été retourné.")
                    df_copie.loc[:, self._list_categorical_columns_with_missing_values] =  array_with_imputed_values
                    return df_copie
                else:
                    logger.info("Le DataFrame avec seulement les colonnes catégorielles dont les valeurs manquantes ont été remplacées a été retourné.")
                    data_partial_with_imputed_values =  pd.DataFrame(array_with_imputed_values, 
                                        columns=self._list_categorical_columns_with_missing_values, index=df_copie.index)
                    return data_partial_with_imputed_values
            else:
                logger.info("Aucune colonne catégorielle avec des valeurs manquantes n'a été détectée.")
                logger.info("Le dataset d'entrée a été retourné sans modification.")
                return data
        except Exception as e:
            logger.exception(e)
    
    @ensure_annotations
    def transform_categorical(self, test_data:pd.DataFrame) -> pd.DataFrame:
        """
        Remplace les valeurs manquantes dans les colonnes catégorielles d'un jeu de données de test.

        Parameters:
            test_data (pd.DataFrame): Le DataFrame contenant les données de test.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes remplacées.
        """
        test_data_ = test_data.copy()
        
        list_categorical_columns_with_missing_values = self._list_categorical_columns_with_missing_values
        try:
            if len(list_categorical_columns_with_missing_values) !=0:
                array_with_imputed_test_values = self.categorical_imputer.transform(test_data_)
                logger.info("Les valeurs manquantes dans les colonnes catégorielles du jeu de données de test ont été imputées.")
                test_data_.loc[:, list_categorical_columns_with_missing_values] =  array_with_imputed_test_values
                logger.info("Le jeu de données de test avec les valeurs manquantes dans les colonnes catégorielles, remplacées a été retourné.")
                return test_data_
            else:
                logger.info("Aucune colonne catégorielle avec des valeurs manquantes n'a été détectée.")
                logger.info("Le dataset de test d'entrée a été retourné sans modification.")
                return test_data
        except Exception as e:
            logger.exception(e)
    
    @ensure_annotations
    def fit_transform_numerical(self, data:pd.DataFrame,  return_whole_dataframe=True) -> pd.DataFrame:
        """
        Remplace les valeurs manquantes dans les colonnes numériques d'un DataFrame.

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.
            return_whole_dataframe (bool): Indique si le DataFrame complet doit être retourné. Par défaut, True.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes remplacées.
        """
        df_copie = data.copy()
        self._list_numerical_columns_with_missing_values = list(self.get_missing_numerical_columns(df_copie).keys())
        try:
            if len(self._list_numerical_columns_with_missing_values) !=0:
                self.numerical_imputer = make_column_transformer((KNNImputer(),
                                                            self._list_numerical_columns_with_missing_values), 
                                                            remainder="drop")
                array_with_imputed_values = self.numerical_imputer.fit_transform(df_copie)
                logger.info("Les valeurs manquantes dans les colonnes numériques ont été imputées.")
                if return_whole_dataframe:
                    logger.info("Le DataFrame complet avec les valeurs manquantes dans les colonnes numériques, remplacées a été retourné.")
                    df_copie.loc[:, self._list_numerical_columns_with_missing_values] =  array_with_imputed_values
                    
                    return df_copie
            
                else:
                    logger.info("Le DataFrame avec seulement les colonnes numériques dont les valeurs manquantes ont été remplacées a été retourné.")
                    data_partial_with_imputed_values =  pd.DataFrame(array_with_imputed_values, 
                                        columns=self._list_numerical_columns_with_missing_values, index=df_copie.index)
                    
                    return data_partial_with_imputed_values
            else:
                logger.info("Aucune colonne numérique avec des valeurs manquantes n'a été détectée.")
                logger.info("Le dataset d'entrée a été retourné sans modification.")
                return data
                
        except Exception as e:
            logger.exception(e)
            
    @ensure_annotations
    def transform_numerical(self, test_data:pd.DataFrame) -> pd.DataFrame:
        """
        Remplace les valeurs manquantes dans les colonnes numériques d'un jeu de données de test.

        Parameters:
            test_data (pd.DataFrame): Le DataFrame contenant les données de test.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes remplacées.
            
        Example:
            # Remplace les valeurs manquantes dans les colonnes numériques du jeu de données de test
            transformed_test_data = IMPUTER.fill_numerical_missing_test_data(test_data)
        """
        test_data_ = test_data.copy()
        
        list_numerical_columns_with_missing_values = self._list_numerical_columns_with_missing_values
        try:
            if len(list_numerical_columns_with_missing_values) !=0:    
                array_with_imputed_test_values = self.numerical_imputer.transform(test_data_)
                logger.info("Les valeurs manquantes dans les colonnes numériques du jeu de données de test ont été imputées.")
                test_data_.loc[:, list_numerical_columns_with_missing_values] =  array_with_imputed_test_values
                logger.info("Le jeu de données de test avec les valeurs manquantes dans les colonnes numériques, remplacées a été retourné.")
                return test_data_
            else:
                logger.info("Aucune colonne numérique avec des valeurs manquantes n'a été détectée.")
                logger.info("Le dataset de test d'entrée a été retourné sans modification.")
                return test_data
        except Exception as e:
            logger.exception(e)
    
    @ensure_annotations
    def fit_transform_pipeline(self, data:pd.DataFrame, return_whole_dataframe=True) -> pd.DataFrame:
        """
        Applique une pipeline d'imputation pour remplacer les valeurs manquantes dans un DataFrame.

        Parameters:
            data (pd.DataFrame): Le DataFrame contenant les données.
            return_whole_dataframe (bool): Indique si le DataFrame complet doit être retourné. Par défaut, True.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes remplacées.

        Example:
            # Retourne le DataFrame complet avec les valeurs manquantes remplacées
            new_data = IMPUTER.impute_missing_values_pipeline(data, return_whole_dataframe=True)
        """
        self.raw_train_data = data.copy()
        full_data = data.copy()
        
        list_numerical_columns_with_missing_values = list(self.get_missing_numerical_columns(full_data).keys())
        list_categorical_columns_with_missing_values = list(self.get_missing_categorical_columns(full_data).keys())
        try:
            if len(list_numerical_columns_with_missing_values) !=0 and\
                len(list_categorical_columns_with_missing_values) !=0:
                self.imputation_pipeline = make_column_transformer((KNNImputer(),
                                                            list_numerical_columns_with_missing_values), 
                                                                (SimpleImputer(strategy="constant", fill_value="missing"), 
                                                                list_categorical_columns_with_missing_values), 
                                                            remainder="drop")
                
                array_with_imputed_values = self.imputation_pipeline.fit_transform(full_data)
                logger.info("Les valeurs manquantes ont été imputées.")
                self._imputed_columns = list_numerical_columns_with_missing_values + list_categorical_columns_with_missing_values
                
                if return_whole_dataframe:
                    full_data.loc[:, self._imputed_columns] = array_with_imputed_values
                    logger.info("Le DataFrame complet avec les valeurs manquantes remplacées a été retourné.")
                    return full_data
                else:
                    data_partial_with_imputed_values =  pd.DataFrame(array_with_imputed_values, 
                                        columns=self._imputed_columns, index=full_data.index)
                    logger.info("Le DataFrame avec seulement les colonnes dont les valeurs manquantes ont été remplacées a été retourné.")
                    return data_partial_with_imputed_values
            else:
                logger.info("Aucune colonne avec des valeurs manquantes n'a été détectée.")
                logger.info("Le dataset d'entrée a été retourné sans modification.")
                return data
        except Exception as e:
            logger.exception(e)
        
    
    @ensure_annotations
    def transform_pipeline(self, test_data:pd.DataFrame) -> pd.DataFrame:
        """
        Remplace les valeurs manquantes dans un jeu de données de test.

        Parameters:
            test_data (pd.DataFrame): Le DataFrame contenant les données de test.

        Returns:
            pd.DataFrame: Le DataFrame avec les valeurs manquantes remplacées.

        Example:
            # Remplace les valeurs manquantes dans les données de test
            transformed_test_data = IMPUTER.fill_missing_test_data(test_data)

        Note:
            Cette fonction utilise les mêmes méthodes d'imputation que celles appliquées sur les données d'entraînement.
        """
        test_data_ = test_data.copy()
        
        
        imputed_columns = self._imputed_columns
        try:
            if len(imputed_columns) !=0 :
                array_with_imputed_test_values = self.imputation_pipeline.transform(test_data_)
                logger.info("Les valeurs manquantes du jeu de données de test ont été imputées.")
                test_data_.loc[:, imputed_columns] =  array_with_imputed_test_values
                logger.info("Le jeu de données de test avec les valeurs manquantes remplacées a été retourné.")
                return test_data_
            else:
                logger.info("Aucune colonne avec des valeurs manquantes n'a été détectée.")
                logger.info("Le dataset de test d'entrée a été retourné sans modification.")
                return test_data
        except Exception as e:
            logger.exception(e)

