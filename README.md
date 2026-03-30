---
language:
- fr
license: mit
tags:
- tabular-regression
- energy
- electricity
- consommation
- logement
- scikit-learn
- joblib
metrics:
- rmse
- r2
---

# ⚡ Prédiction de la Consommation Électrique d'un Logement

## Description

Ce modèle prédit la **consommation électrique annuelle d'un logement** (en kWh) à partir de ses caractéristiques. Il peut aider les propriétaires, locataires ou professionnels de l'immobilier à estimer les charges énergétiques d'un bien avant occupation.

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

## Performances

| Métrique | Valeur |
|----------|--------|
| RMSE | À compléter |
| R² | À compléter |

## Limites

- Les comportements individuels des occupants ont un impact fort non modélisé
- Le modèle ne tient pas compte des équipements spécifiques (piscine, borne de recharge, etc.)
- La précision est meilleure sur les logements standards

## Auteur

Développé par [a126OPS](https://huggingface.co/a126OPS)  
🔗 Démo interactive : [conso_energie_predict](https://huggingface.co/spaces/a126OPS/conso_energie_predict)

## Licence

[MIT](https://opensource.org/licenses/MIT)
