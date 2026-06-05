# Déploiement — pas à pas

## 1. Créer le dépôt GitHub (vide)

1. Va sur <https://github.com/new>
2. **Repository name** : `fraud-detection-dashboard`
3. Visibilité : **Public** (requis pour le plan gratuit Streamlit Cloud)
4. **NE PAS** cocher « Add a README », « .gitignore » ni « license » (ils existent déjà ici)
5. Clique **Create repository**

## 2. Pousser le code

Dans PowerShell :

```powershell
cd C:\Users\aella\02_PORTFOLIO_ABDOULLATUF\fraud-detection-dashboard
git remote add origin https://github.com/Abdoullatuf/fraud-detection-dashboard.git
git push -u origin main
```

(Une fenêtre de connexion GitHub peut s'ouvrir la première fois.)

## 3. Déployer sur Streamlit Community Cloud

1. Va sur <https://share.streamlit.io> → **Sign in with GitHub**
2. **Create app** → **Deploy a public app from GitHub**
3. Renseigne :
   - **Repository** : `Abdoullatuf/fraud-detection-dashboard`
   - **Branch** : `main`
   - **Main file path** : `streamlit_app.py`
4. **Advanced settings** → **Python version** : `3.12`
5. (Optionnel mais recommandé) **App URL** : choisis le sous-domaine `fraud-detection-dashboard`
   → l'URL finale sera `https://fraud-detection-dashboard.streamlit.app/`
6. **Deploy** et attends la fin du build (2–4 min).

## 4. Si l'URL diffère

Si Streamlit ajoute un hash (ex. `fraud-detection-dashboard-xxxx.streamlit.app`), mets à jour le lien dans :
- `README.md` (section « Application en ligne »)
- `../abdoullatuf.github.io/index.html` (bouton « Dashboard en ligne » de la carte)

## 5. Publier la carte sur le portfolio

```powershell
cd C:\Users\aella\02_PORTFOLIO_ABDOULLATUF\abdoullatuf.github.io
git push
```

GitHub Pages republie le site automatiquement.
