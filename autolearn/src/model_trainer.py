import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path

from sklearn.linear_model import ( BayesianRidge, Ridge, ElasticNet, Lasso, ARDRegression)
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, 
                        GradientBoostingRegressor, HistGradientBoostingRegressor, 
                              ExtraTreesRegressor, IsolationForest)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (make_scorer, mean_absolute_error,
                             median_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import learning_curve, cross_validate
from  sklearn.model_selection import KFold

from .logger import logger

import warnings
warnings.filterwarnings("ignore")

def root_mean_squared_error_scorer(y, y_pred):
    
    return np.sqrt(mean_squared_error(y, y_pred))

class ModelTrainer:
    def __init__(self):
        pass
    
    def compare_models(self, X, y):
        """
        Compare les performances de plusieurs modèles de machine learning sur un ensemble de données.

        Parameters:
            X (np.ndarray): Le DataFrame contenant les variables indépendantes.
            y (pd.Series): La Series contenant la variable cible.

        Returns:
            pd.DataFrame: Un DataFrame contenant les performances des modèles comparés.
            
        Note:
            Cette fonction utilise une méthode de validation croisée pour évaluer les performances des modèles.
        """
        scoring = {
        'RMSE': make_scorer(root_mean_squared_error_scorer, greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'MedianAE': make_scorer(median_absolute_error, greater_is_better=False), 
        'R2': make_scorer(r2_score)}
        
        self.models = {
            "BayesianRidge" : BayesianRidge(), 
            "Ridge" : Ridge(), 
            "ElasticNet" : ElasticNet(), 
            "Lasso" : Lasso(), 
            "RandomForestRegressor" : RandomForestRegressor(), 
            "AdaBoostRegressor" :  AdaBoostRegressor(), 
            "GradientBoostingRegressor" : GradientBoostingRegressor(), 
            "HistGradientBoostingRegressor" : HistGradientBoostingRegressor(), 
            "ExtraTreesRegressor" : ExtraTreesRegressor(), 
            "KNeighborsRegressor" : KNeighborsRegressor(), 
            "XGBRegressor" : XGBRegressor(verbose=False), 
            "CatBoostRegressor" : CatBoostRegressor(verbose=False), 
            "LGBMRegressor" : LGBMRegressor(verbose=-1), 
            "IsolationForest" : IsolationForest(), 
            "ARDRegression" : ARDRegression()
            }
        logger.info(f"Nombre de modèles à comparer : {len(self.models)}.")
        model_metrics = []
        self.learning_curve_scores = {}
        results = {}
        kf = KFold(n_splits=10)
        train_sizes = np.linspace(0.1, 1.0, num=10)
        try:
            for name, model in tqdm(self.models.items()):
                logger.info(f"Validation croisée en cours pour le modèle : {name}.")
                self.learning_curve_scores[name] = {}
                results[name] = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)
                name_mean_metrics = (name, -results[name]["test_RMSE"].mean(), 
                                    -results[name]["test_MAE"].mean(), 
                                    -results[name]["test_MedianAE"].mean(), 
                                    results[name]["test_R2"].mean())
                model_metrics.append(name_mean_metrics)
                
                N, train_score, val_score = learning_curve(model, X, y, cv= kf,
                                                    train_sizes=train_sizes, scoring="neg_mean_squared_error")
                
                self.learning_curve_scores[name]["mean_train_score"] = -train_score.mean(axis=1)
                self.learning_curve_scores[name]["mean_val_score"] = -val_score.mean(axis=1)
                self.N = N
            col_formatters = ["RMSE", "MAE", "MedianAE"]
            self.model_metrics_data = pd.DataFrame(model_metrics, columns=["Name", "RMSE", "MAE", "MedianAE", "R2"])
            self.model_metrics_data =  self.model_metrics_data.sort_values(by=["RMSE", "MAE", "MedianAE"], ascending=True)\
                                                            .sort_values(by="R2", ascending=False)
            #artefacts_dir = Path("./auto_learn/artefacts")
            #os.makedirs(artefacts_dir, exist_ok=True)
            #self.model_metrics_data.to_csv(f"{artefacts_dir}/model_performance.csv", index=False)
            #logger.info("Le DataFrame contenant les performances des modèles a été enregistré.")
                                                        
            self.best_estimator_ = self.models[self.model_metrics_data.iloc[0, 0]]
            logger.info(f"Le meilleur modèle identifié : {self.model_metrics_data.iloc[0, 0]}.")

            return self.model_metrics_data\
                                        .style.highlight_min(subset=col_formatters, color="green")\
                                        .highlight_max(subset=["R2"], color="green")
        except Exception as e:
            logger.exception(e)
    
    def plot_learning_curves(self):
        """
        Affiche les courbes d'apprentissage pour les modèles entraînés.

        Returns:
            None
        """
        try:
            plt.figure(figsize=(30,15))
            i=1
            for name, scores in self.learning_curve_scores.items():
                plt.subplot(3, 5, i)
                plt.plot(self.N, scores["mean_train_score"], 'o-', color="blue", label="Training score")
                plt.plot(self.N, scores["mean_val_score"], 'o-', color="darkorange", label="validation score")
                plt.title(f"{name}")
                plt.xlabel("Samples")
                plt.ylabel("Mean Squared Error")
                plt.legend()
                i += 1
            #artefacts_dir = Path("./auto_learn/artefacts")
            #os.makedirs(artefacts_dir, exist_ok=True)
            #plt.savefig(f"{artefacts_dir}/learning_curves.png")
            #logger.info("Les courbes d'apprentissage des modèles ont été enregistrées.")
            plt.show()
        except Exception as e:
            logger.exception(e)
        

