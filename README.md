# Prédiction du revenu à partir des données de recensement

## Objectif
Ce projet vise à prédire la classe de revenu `Income` à partir des données de recensement
en utilisant des méthodes de classification supervisée.

La variable cible est binaire :
- 0 : revenu > 50K
- 1 : revenu ≤ 50K

Les méthodes d’ensemble (Bagging et Boosting) ainsi que la validation croisée sont mises en œuvre
conformément aux exigences du semestre S8.

## Jeu de données
Le jeu de données `census.csv` contient des variables numériques et catégorielles issues d’un recensement.

## Méthodologie
- Analyse exploratoire (shape, info, describe, head, value_counts)
- Visualisations (pairplot, régressions)
- Prétraitement avec pipelines sklearn
- Modèles : KNN, Random Forest (Bagging), AdaBoost (Boosting)
- Validation croisée (GridSearchCV)

## Déploiement
Le modèle final est sauvegardé sous le nom :
recensement.pkl

## Exécution
pip install -r requirements.txt
python train_census.py

## Auteur
Projet académique – Machine Learning S8
