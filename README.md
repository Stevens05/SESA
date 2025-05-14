# Projet de machine learning sur le **Sound Events for Surveillance Applications**
  
Le dataset Sound Events for Surveillance Applications (SESA) a été obtenu auprès de Freesound. L'ensemble des données a été divisé entre les dossiers « train » (480 fichiers) et « test » (105 fichiers). Tous les fichiers audio sont de type WAV, mono-canal, 16 kHz et 8 bits, avec une durée maximale de 33 secondes. # Classes: 0 - Casual (not a threat) 1 - Gunshot 2 - Explosion 3 - Siren (also contains alarms)

Ce projet de machine learning comprend un dashboard interactif développé avec **FastAPI** et une API déployée sur **Microsoft Azure**. L'objectif de ce projet est de fournir des prédictions basées sur un modèle de machine learning entraîné.


## Partie machine learning (`model` folder)
  
Cette partie est composé du notebook `Train_built-in.ipynb` qui contient :
- le preprocessing et l'extraction des features dans les fichiers Audio
- la conception de différents modèles de ML pour la classification des sons : **Régression logistique**, **K-NN**, **Random Forest**, **SVM**, **XGBoost**
- l'enrégistrement du meilleur modèle `best_multiclass_model.pkl`
 modélisation ainsi que tous les fichiers qui seront utilisés dans la seconde partie (API et Dashboard)

## Partie MLOps
- Conception de l'API via **FastAPI**
- Containeurisation de l'API
- Déploiement sur **Microsoft Azure**
- 

## Structure du projet

Le projet est organisé comme suit :

- data
    - contient les données d'entrainement et de test

- model
    - `best_multiclass_model.pkl` : 
    - `label_encoder.pkl` : 
    - `scaler.pkl` : 

- k8s

- utils
    - `audio_features.py` : contient la fonction permettant l'extraction des features à partir des fichiers audio

- `main.py` : contient l'API FastAPI
- `requirements.txt` dépendances python à installer