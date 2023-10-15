# Auto-Learn

Le package `autolearn` est une bibliothèque d'apprentissage automatique conçue pour automatiser le processus d'apprentissage automatique. Son objectif principal est de simplifier et d'accélérer la création de modèles prédictifs tout en minimisant la quantité de code nécessaire. Spécialement conçu pour résoudre des problèmes de régression, il offre différentes stratégies dédiées à la préparation de données.

Avec `autolearn`, nous pouvons automatiser des étapes cruciales de la préparation de données, telles que l'imputation des valeurs manquantes. De plus, il permet d'appliquer des transformations sur les variables numériques, comme la log transformation, afin d'ajuster leur distribution vers une forme plus normale. Une autre fonctionnalité fondamentale est la standardisation, qui ramène les valeurs des variables à une plage fixe de [0, 1]. Cela s'avère essentiel pour les modèles paramétriques, tels que les modèles linéaires, en facilitant leur convergence et en optimisant le calcul des distances pour évaluer la similarité entre deux individus, comme les modèles de K-Nearest Neighbors.

En ce qui concerne les variables catégorielles, `autolearn` adopte une approche différenciée. Il applique le one-hot-encoding pour certaines d'entre elles, tandis que pour d'autres, il effectue une transformation qui consiste à remplacer chaque modalité par sa fréquence. Cette dernière transformation, particulièrement cruciale pour les variables comportant de nombreuses modalités, évite une explosion du nombre de variables pendant l'entraînement des modèles.

Enfin, une fois les données correctement préparées, il permet d'entraîner ces données sur une variété d'algorithmes de machine learning. Il compare ensuite leurs performances, permettant de sélectionner le meilleur modèle pour faire des prédictions sur les données de test.

Ce projet s'appuie sur les expériences et compétences que j'ai acquises en machine learning et en statistique à l'ISUP. `autolearn` vise à faciliter considérablement le processus de création de modèles prédictifs avec un minimum d'efforts.

## Fonctionnalités

Les fonctionnalités clés du package comprennent :

<p style="color: blue"> <B>Imputation des valeurs manquantes </B>:</p> 

Il permet d'automatiser l'imputation des valeurs manquantes dans les données.

<p style="color: blue"> <B> Transformation des Variables Numériques </B>:</p> 

Il permet d'appliquer des transformations telles que la log-transformation pour normaliser la distribution des variables. De plus, il effectue une standardisation pour ramener les valeurs dans une plage fixe de [0, 1].

<p style="color: blue"> <B>Gestion des Variables Catégorielles </B>:</p> 

Il adopte une approche différenciée. Il applique le one-hot-encoding pour certaines variables catégorielles, tandis que pour d'autres, il remplace chaque modalité par sa fréquence. 

<p style="color: blue"> <B>Entraînement de Modèles</B>:</p> 

Une fois les données correctement préparées,`autolearn` permet d'entraîner ces données sur une variété d'algorithmes de machine learning.

<p style="color: blue"> <B>Comparaison des Modèles</B>:</p>

Il compare ensuite leurs performances, facilitant ainsi la sélection du meilleur modèle pour les prédictions sur les données de test.

## Configuration

Pour configurer et utiliser `autolearn`, suivez les étapes ci-dessous :

1. **Création d'un dossier :** Créez un dossier sur votre système où vous souhaitez stocker le projet `autolearn`.

2. **Création de l'environnement virtuel :** Ouvrez un terminal et naviguez vers le dossier que vous venez de créer. Ensuite, exécutez la commande suivante pour créer un environnement virtuel:

   ```bash
   python -m venv .venv
   ```
   - Sur Windows
   ```bash
   .\env\Scripts\activate
   ```
   - Sur macOS et Linux
   ```bash
   source env/bin/activate
   ```
3. **Construction et Installation :** Ouvrez un terminal, naviguez vers le dossier où il est stocké et exécutez les commandes suivantes :
   ```bash
   pip install wheel
   python setup.py bdist_wheel
   python setup.py sdist
   pip install .
   ```
## Démonstration
Voici quelques exemples pour vous montrer comment utiliser `autolearn` :
### Imputation des valeurs manquantes : 
```python
from autolearn import DataImputer
imputer = DataImputer()

# Imputez les valeurs manquantes dans un DataFrame (trainset, testset)
imputed_df_train = imputer.fit_transform_pipeline(df_train)
imputed_df_test = imputer.transform_pipeline(df_test)
```
### Transformation des données : 
```python
from autolearn import DataTransformer

transformer = DataTransformer(alpha=0.05, threshold=5)

# Obtenez les données transformées(trainset, testset)
X_train = transformer.fit_transform(imputed_df_train)
X_test= transformer.transform(imputed_df_test)
```

### Entraînement des modèles : 
```python
from autolearn import ModelTrainer

trainer = ModelTrainer()

# Comparez les performances des modèles
model_comparison_results = trainer.compare_models(X_train, y_train)
# Tracer les courbes d'apprentissage des modèles
trainer.plot_learning_curves()
```

### Prédiction : 
```python
# Prédiction sur les données de test
y_pred = trainer.best_estimator_.predict(X_test)
```
## Résultats : 
### Courbes d'apprentissage
![Courbes d'apprentissage](./auto_learn/artefacts/learning_curves.png)
## Licence
Ce projet est sous licence MIT. Pour plus d'informations, veuillez consulter le fichier [MIT](./LICENSE).

