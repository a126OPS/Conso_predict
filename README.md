# Conso Predict

Ce projet contient le notebook `conso_electrique_france.ipynb`.
Il estime la consommation electrique residentielle en France a partir des donnees ouvertes de l'Agence ORE / ENEDIS, entraine plusieurs modeles ML, explique le resultat avec SHAP, puis propose une interface Gradio.

Le notebook a ete cree avec un noyau Python 3.10, et il a ete verifie ici avec Python 3.12.6 le 22/03/2026.

## Prerequis

- Windows
- Python 3.12 installe
- Connexion Internet pour installer les dependances
- Connexion Internet pendant l'execution du notebook pour telecharger les donnees ORE

## Demarrage rapide

Dans PowerShell, depuis le dossier du projet :

```powershell
py -3.12 -m venv conso
.\conso\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name conso --display-name "Python (conso)"
jupyter lab
```

Si le dossier `conso` existe deja, vous pouvez simplement l'activer et ignorer la ligne `py -3.12 -m venv conso`.

Ensuite :

1. Ouvrez `conso_electrique_france.ipynb`
2. Selectionnez le kernel `Python (conso)`
3. Executez les cellules dans l'ordre

## Si l'activation PowerShell est bloquee

Executez une seule fois :

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Puis relancez :

```powershell
.\conso\Scripts\Activate.ps1
```

## Ce que produit le notebook

Pendant l'execution, le notebook :

- telecharge les donnees ENEDIS / Agence ORE
- prepare les variables metier
- entraine `Ridge`, `Random Forest` et `XGBoost`
- affiche des graphiques et des explications SHAP
- sauvegarde `modele_conso_elec.joblib`
- construit une interface `Gradio`

## Important pour la derniere cellule

La derniere cellule lance :

```python
demo.launch(share=True, debug=True)
```

Cela ouvre l'interface Gradio et peut creer un lien public temporaire.

Si vous voulez seulement executer l'analyse et l'entrainement du modele :

- arretez-vous avant la derniere cellule
- ou remplacez `share=True` par `share=False`

## Verification rapide

Une fois l'environnement `conso` active :

```powershell
python -c "import pandas, numpy, requests, joblib, matplotlib, seaborn, sklearn, xgboost, shap, gradio, jupyterlab, notebook, ipykernel; print('OK')"
```

## Validation realisee ici

Le notebook a ete teste dans l'environnement `conso` avec :

- chargement des donnees ORE
- preparation du dataset
- entrainement des modeles
- calcul SHAP
- generation du fichier `modele_conso_elec.joblib`

Le lancement automatique de Gradio n'a pas ete execute pendant le test pour eviter de bloquer la session sur le serveur web local.

## Fichiers utiles

- `conso_electrique_france.ipynb` : notebook principal
- `requirements.txt` : dependances Python a installer
- `conso/` : environnement virtuel local du projet
- `app.py` : fichier Python actuellement vide
