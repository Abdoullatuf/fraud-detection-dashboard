# Détection de Fraude & Rentabilité

Dashboard interactif de détection de fraude sur transactions e-commerce, optimisé non pas sur l'exactitude mais sur le **profit net** de la stratégie de blocage. Projet réalisé dans le cadre de l'UE **STA218 — Science de la donnée en milieu professionnel** (M2 Science des Données, CNAM).

## Application en ligne

▶️ **[Lancer le dashboard](https://fraud-detection-dashboard-ml.streamlit.app/)**

## Résultat clé

Le modèle retenu (**XGBoost**, seuil de décision calibré à **0,66** par maximisation du profit sur la validation) dégage un **profit net de 46 074 $ sur l'échantillon de test**, soit **+9,8 %** par rapport à une stratégie « accepter toutes les transactions », pour un taux de blocage opérationnel de **3,6 %**.

| Métrique | Validation | Test |
|---|---:|---:|
| PR-AUC | 0,142 | 0,146 |
| ROC-AUC | 0,622 | 0,652 |
| Profit @ 0,66 | 48 310 $ | 46 074 $ |
| Recall fraude | — | 20,2 % |
| Précision des blocages | — | 25,6 % |
| Taux de blocage | — | 3,6 % |

## Fonctionnalités

- **Prédiction en direct** : scoring d'une transaction saisie au formulaire ou en JSON.
- **Explication de la décision** : raisons métier + valeurs SHAP locales.
- **Seuil de profit interactif** : curseur pour explorer l'arbitrage blocage / profit.
- **Performances du modèle** : courbes ROC & PR, matrice de confusion.
- **Prédiction par lot** : scoring d'un fichier CSV complet.

## Stack technique

- **Modèle** : XGBoost (classification binaire, seuil calibré sur le profit)
- **Frontend** : Streamlit
- **Visualisation** : Altair, Matplotlib
- **Interprétabilité** : SHAP
- **Données** : Fraud_Data (151 112 transactions e-commerce) + géolocalisation IP

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Structure du projet

```
fraud-detection-dashboard/
├── streamlit_app.py                 # Application Streamlit (6 onglets)
├── requirements.txt                 # Dépendances Python (versions épinglées)
├── scripts/
│   └── demo_prediction_live.py      # Logique de prédiction + raisons métier
├── outputs/
│   └── model/
│       ├── fraud_profit_model.joblib    # Modèle XGBoost + preprocessor + seuil
│       ├── dashboard_cache.joblib       # Prédictions valid/test (alimente le dashboard)
│       └── sample_transactions.json     # Exemples fraude / légitime
└── notebooks/
    └── sta218_projet_fraude_rentabilite.ipynb   # Analyse complète (EDA → modélisation → profit)
```

## Auteur

**Maoulida Abdoullatuf** — M2 Science des Données, CNAM · Data Scientist certifié (RNCP niveau 7)
