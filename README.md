---
title: Conso Energie Predict
sdk: gradio
app_file: app.py
python_version: "3.12"
sdk_version: "5.23.1"
pinned: false
---

# Conso Predict

Ce projet estime la consommation electrique residentielle en France a partir des donnees ouvertes de l'Agence ORE / ENEDIS et propose une interface Gradio de prediction.

Le depot contient :

- `app.py` : interface Gradio autonome
- `modele_conso_elec.joblib` : modele exporte
- `conso_electrique_france.ipynb` : notebook d'analyse et d'entrainement

## Lancer localement

Dans PowerShell, depuis le dossier du projet :

```powershell
py -3.12 -m venv conso
.\conso\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

L'application choisit automatiquement un port libre en local.

## Lancer le notebook

Si vous voulez reproduire l'entrainement :

```powershell
python -m ipykernel install --user --name conso --display-name "Python (conso)"
jupyter lab
```

Puis ouvrez `conso_electrique_france.ipynb`, selectionnez `Python (conso)` et executez les cellules dans l'ordre.

## Si l'activation PowerShell est bloquee

Executez une seule fois :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Puis relancez :

```powershell
.\conso\Scripts\Activate.ps1
```

## Verification rapide

Une fois l'environnement `conso` active :

```powershell
python -c "import gradio, joblib, numpy, pandas, sklearn, xgboost; print('OK')"
```

## Validation realisee ici

L'application Gradio et le chargement du modele ont ete verifies avec Python 3.12.6.

## Fichiers utiles

- `app.py` : interface Gradio principale
- `modele_conso_elec.joblib` : artefact du modele
- `conso_electrique_france.ipynb` : notebook principal
- `requirements.txt` : dependances Python
