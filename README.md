---
title: Conso Energie Predict
emoji: "⚡"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ⚡ Prédiction de la Consommation Électrique d'un Logement

## Description

Ce modèle prédit la **consommation électrique annuelle d'un logement** (en kWh) à partir de ses caractéristiques. Il peut aider les propriétaires, locataires ou professionnels de l'immobilier à estimer les charges énergétiques d'un bien avant occupation.

## API portfolio

Un dossier [`api`](./api) a été ajouté pour exposer le projet via FastAPI et permettre son intégration dans un portfolio.

Lancement local :

```bash
conso\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints utiles :

- `GET /api/project` : métadonnées du projet et URLs d'intégration
- `GET /api/departments` : liste des départements pour alimenter un formulaire côté portfolio
- `POST /api/predict` : endpoint JSON de prédiction
- `GET /embed` : page d'embed prête pour une `iframe`
- `GET /interface` : interface Gradio montée dans l'API

Variables d'environnement optionnelles :

- `PUBLIC_BASE_URL` : URL publique à renvoyer dans les endpoints
- `PORTFOLIO_ORIGINS` : origines CORS autorisées, séparées par des virgules
- `API_HOST` et `API_PORT` : hôte et port du serveur API

## Utilisation

```python
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Chargement du modèle
model_path = hf_hub_download(repo_id="a126OPS/conso_energie_predict", filename="model.joblib")
model = joblib.load(model_path)

# Exemple de prédiction
# [surface_m2, nb_occupants, type_chauffage, annee_construction, type_logement]
features = np.array([[75, 2, 1, 1985, 0]])  # 75m², 2 personnes, électrique, 1985, appartement
predicted_conso = model.predict(features)
print(f"Consommation estimée : {predicted_conso[0]:.0f} kWh/an")
```

## Données d'entraînement

- **Source :** Données de performance énergétique des logements (DPE) — ADEME / data.gouv.fr
- **Variables d'entrée :** surface habitable, nombre d'occupants, type de chauffage, année de construction, type de logement (maison / appartement)
- **Variable cible :** consommation électrique en kWh/an

## Limites

- Les comportements individuels des occupants ont un impact fort non modélisé
- Le modèle ne tient pas compte des équipements spécifiques (piscine, borne de recharge, etc.)
- La précision est meilleure sur les logements standards

## Auteur

Développé par [a126OPS](https://huggingface.co/a126OPS)  
🔗 Démo interactive : [conso_energie_predict](https://huggingface.co/spaces/a126OPS/conso_energie_predict)

## Licence

[MIT](https://opensource.org/licenses/MIT)
