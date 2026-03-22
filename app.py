from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import gradio as gr
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modele_conso_elec.joblib"

TARIF_KWH = 0.2516
ABONNEMENT_AN = 120
CONSO_REF_NAT = 4819
CONSO_REF_APPART = 5500
CONSO_REF_MAISON = 8000

NOMS_DEPTS = {
    "1": "Ain",
    "2": "Aisne",
    "3": "Allier",
    "4": "Alpes-de-Haute-Provence",
    "5": "Hautes-Alpes",
    "6": "Alpes-Maritimes",
    "7": "Ardeche",
    "8": "Ardennes",
    "9": "Ariege",
    "10": "Aube",
    "11": "Aude",
    "12": "Aveyron",
    "13": "Bouches-du-Rhone",
    "14": "Calvados",
    "15": "Cantal",
    "16": "Charente",
    "17": "Charente-Maritime",
    "18": "Cher",
    "19": "Correze",
    "21": "Cote-d'Or",
    "22": "Cotes-d'Armor",
    "23": "Creuse",
    "24": "Dordogne",
    "25": "Doubs",
    "26": "Drome",
    "27": "Eure",
    "28": "Eure-et-Loir",
    "29": "Finistere",
    "30": "Gard",
    "31": "Haute-Garonne",
    "32": "Gers",
    "33": "Gironde",
    "34": "Herault",
    "35": "Ille-et-Vilaine",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "38": "Isere",
    "39": "Jura",
    "40": "Landes",
    "41": "Loir-et-Cher",
    "42": "Loire",
    "43": "Haute-Loire",
    "44": "Loire-Atlantique",
    "45": "Loiret",
    "46": "Lot",
    "47": "Lot-et-Garonne",
    "48": "Lozere",
    "49": "Maine-et-Loire",
    "50": "Manche",
    "51": "Marne",
    "52": "Haute-Marne",
    "53": "Mayenne",
    "54": "Meurthe-et-Moselle",
    "55": "Meuse",
    "56": "Morbihan",
    "57": "Moselle",
    "58": "Nievre",
    "59": "Nord",
    "60": "Oise",
    "61": "Orne",
    "62": "Pas-de-Calais",
    "63": "Puy-de-Dome",
    "64": "Pyrenees-Atlantiques",
    "65": "Hautes-Pyrenees",
    "66": "Pyrenees-Orientales",
    "67": "Bas-Rhin",
    "68": "Haut-Rhin",
    "69": "Rhone",
    "70": "Haute-Saone",
    "71": "Saone-et-Loire",
    "72": "Sarthe",
    "73": "Savoie",
    "74": "Haute-Savoie",
    "75": "Paris",
    "76": "Seine-Maritime",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "79": "Deux-Sevres",
    "80": "Somme",
    "81": "Tarn",
    "82": "Tarn-et-Garonne",
    "83": "Var",
    "84": "Vaucluse",
    "85": "Vendee",
    "86": "Vienne",
    "87": "Haute-Vienne",
    "88": "Vosges",
    "89": "Yonne",
    "90": "Territoire de Belfort",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
    "2A": "Corse-du-Sud",
    "2B": "Haute-Corse",
}


def dept_sort_key(code: str) -> tuple[int, str]:
    try:
        return (0, f"{int(code):03d}")
    except ValueError:
        return (1, code)


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            "Le fichier modele_conso_elec.joblib est introuvable. "
            "Executez d'abord le notebook pour generer le modele."
        )
    return joblib.load(model_path)


ARTIFACT = load_artifact()
PIPELINE = ARTIFACT["pipeline"]
FEATURES = ARTIFACT["features"]
DF_MODEL = ARTIFACT["df_model"].copy()
MAE = ARTIFACT.get("mae", 307)
R2 = ARTIFACT.get("r2", 0.42)

DEPTS_DISPO = sorted(DF_MODEL["dept"].astype(str).unique(), key=dept_sort_key)
CHOIX_DEPTS = []
for dept in DEPTS_DISPO:
    try:
        code_fmt = str(int(dept)).zfill(2)
    except ValueError:
        code_fmt = dept
    nom = NOMS_DEPTS.get(dept, NOMS_DEPTS.get(code_fmt, dept))
    CHOIX_DEPTS.append(f"{code_fmt} - {nom}")


def estimer_conso_foyer(
    dept: str,
    type_logement: str,
    surface_m2: int,
    nb_personnes: int,
    annee_construction: int,
    chauffage_electrique: bool,
    annee: int = 2024,
) -> dict:
    ligne = DF_MODEL[
        (DF_MODEL["dept"].astype(str) == str(dept)) & (DF_MODEL["annee"] == annee)
    ]
    if ligne.empty:
        ligne = DF_MODEL[DF_MODEL["dept"].astype(str) == str(dept)].sort_values(
            "annee", ascending=False
        ).head(1)

    if not ligne.empty:
        conso_base = float(PIPELINE.predict(ligne[FEATURES])[0])
        facteur_dept = max(0.85, min(1.30, conso_base / CONSO_REF_NAT))
    else:
        conso_base = float(CONSO_REF_NAT)
        facteur_dept = 1.0

    conso_ref = CONSO_REF_MAISON if type_logement == "Maison" else CONSO_REF_APPART
    surf_ref = 90 if type_logement == "Maison" else 65
    ratio_surface = (surface_m2 / surf_ref) ** 0.75
    ratio_chauffage = 1.65 if chauffage_electrique else 1.0

    if annee_construction >= 2012:
        ratio_isolation = 0.65
    elif annee_construction >= 2000:
        ratio_isolation = 0.80
    elif annee_construction >= 1990:
        ratio_isolation = 0.92
    elif annee_construction >= 1975:
        ratio_isolation = 1.00
    elif annee_construction >= 1960:
        ratio_isolation = 1.05
    else:
        ratio_isolation = 1.10

    pers_ref = 3 if type_logement == "Maison" else 2
    ratio_personnes = (nb_personnes / pers_ref) ** 0.45

    conso_estimee = (
        conso_ref
        * facteur_dept
        * ratio_surface
        * ratio_chauffage
        * ratio_isolation
        * ratio_personnes
    )
    conso_estimee = max(1000, min(22000, conso_estimee))

    facture_annuelle = conso_estimee * TARIF_KWH + ABONNEMENT_AN
    facture_mensuelle = facture_annuelle / 12

    return {
        "conso_base_dept": round(conso_base),
        "facteur_dept": round(facteur_dept, 2),
        "conso_estimee_kwh": round(conso_estimee),
        "facture_annuelle": round(facture_annuelle),
        "facture_mensuelle": round(facture_mensuelle),
    }


def scorer_conso(conso_kwh: int, surface_m2: int, type_logement: str) -> tuple[str, float, dict]:
    conso_m2 = conso_kwh / surface_m2
    refs = {
        "Appartement": {"econome": 40, "normal": 80},
        "Maison": {"econome": 60, "normal": 110},
    }.get(type_logement, {"econome": 40, "normal": 80})

    if conso_m2 <= refs["econome"]:
        score = "econome"
    elif conso_m2 <= refs["normal"]:
        score = "normal"
    else:
        score = "energivore"
    return score, round(conso_m2, 1), refs


def estimer_interface(
    dept_str: str,
    type_logement: str,
    surface_m2: float,
    nb_personnes: float,
    annee_construction: float,
    chauffage_elec: str,
) -> str:
    dept_raw = dept_str.split(" - ")[0].strip()
    try:
        dept = str(int(dept_raw))
    except ValueError:
        dept = dept_raw

    resultat = estimer_conso_foyer(
        dept=dept,
        type_logement=type_logement,
        surface_m2=int(surface_m2),
        nb_personnes=int(nb_personnes),
        annee_construction=int(annee_construction),
        chauffage_electrique=chauffage_elec == "Oui",
    )

    conso = resultat["conso_estimee_kwh"]
    score, conso_m2, refs = scorer_conso(conso, int(surface_m2), type_logement)
    couleur = {"econome": "Vert", "normal": "Jaune", "energivore": "Rouge"}[score]
    label = {
        "econome": "ECONOME",
        "normal": "DANS LA MOYENNE",
        "energivore": "ENERGIVORE",
    }[score]

    conseils = []
    if int(annee_construction) < 1975:
        conseils.append("Logement ancien : l'isolation est prioritaire.")
    if chauffage_elec == "Oui" and score == "energivore":
        conseils.append("Le chauffage electrique peut peser fort sur la facture.")
    if score == "energivore":
        conseils.append("LED, veille coupee et reglages de chauffage peuvent reduire la conso.")
    if not conseils:
        conseils.append("Consommation raisonnable pour ce type de logement.")

    nom_dept = NOMS_DEPTS.get(dept, dept)
    conseils_md = "\n".join(f"- {conseil}" for conseil in conseils)

    return f"""
## Estimation pour {type_logement} {int(surface_m2)} m2 - {nom_dept}

| Consommation estimee | Facture mensuelle | Facture annuelle |
|:---:|:---:|:---:|
| **{conso:,} kWh/an** | **{resultat["facture_mensuelle"]} EUR/mois** | **{resultat["facture_annuelle"]} EUR/an** |

### Score : {couleur} - {label}
_{conso_m2:.0f} kWh/m2/an_
_(reference : < {refs["econome"]} kWh/m2 econome, > {refs["normal"]} kWh/m2 energivore)_

### Votre logement
| | |
|:---|:---|
| Departement | {nom_dept} |
| Type | {type_logement} |
| Surface | {int(surface_m2)} m2 |
| Personnes | {int(nb_personnes)} |
| Construction | {int(annee_construction)} |
| Chauffage elec | {chauffage_elec} |
| Facteur local | x{resultat["facteur_dept"]} vs moyenne nationale |

### Conseils
{conseils_md}

---
*Tarif EDF 2024 : {TARIF_KWH} EUR/kWh - Abonnement : {ABONNEMENT_AN} EUR/an*  
*Precision : +/- {MAE} kWh/an - estimation indicative*  
*Performance du modele : R2 = {R2}*
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Estimation Consommation Electrique",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            f"""
# Estimateur de Consommation Electrique
## France - Donnees ENEDIS officielles 2015-2024
> Base : **{len(DF_MODEL):,} observations** (94 departements x 10 ans)  
> Source : Agence ORE / data.gouv.fr
"""
        )

        with gr.Row():
            with gr.Column():
                dept_input = gr.Dropdown(
                    choices=CHOIX_DEPTS,
                    value="75 - Paris",
                    label="Departement",
                )
                type_input = gr.Radio(
                    choices=["Appartement", "Maison"],
                    value="Appartement",
                    label="Type de logement",
                )
            with gr.Column():
                surface_input = gr.Slider(10, 300, value=65, step=5, label="Surface (m2)")
                pers_input = gr.Slider(1, 6, value=2, step=1, label="Nombre de personnes")
                annee_input = gr.Slider(
                    1900, 2024, value=1990, step=1, label="Annee de construction"
                )
                chauffage_input = gr.Radio(
                    choices=["Non", "Oui"],
                    value="Non",
                    label="Chauffage electrique ?",
                )

        btn = gr.Button("Estimer ma consommation", variant="primary", size="lg")
        output = gr.Markdown("_Remplissez le formulaire puis cliquez sur Estimer._")

        btn.click(
            fn=estimer_interface,
            inputs=[
                dept_input,
                type_input,
                surface_input,
                pers_input,
                annee_input,
                chauffage_input,
            ],
            outputs=output,
        )

        gr.Markdown("### Exemples")
        gr.Examples(
            examples=[
                ["75 - Paris", "Appartement", 45, 1, 1990, "Non"],
                ["69 - Rhone", "Maison", 100, 4, 1975, "Oui"],
                ["71 - Saone-et-Loire", "Appartement", 50, 2, 2015, "Non"],
                ["13 - Bouches-du-Rhone", "Maison", 90, 3, 2005, "Non"],
                ["67 - Bas-Rhin", "Maison", 120, 4, 1965, "Oui"],
            ],
            inputs=[
                dept_input,
                type_input,
                surface_input,
                pers_input,
                annee_input,
                chauffage_input,
            ],
        )

    return demo


def running_on_spaces() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SYSTEM") == "spaces")


def find_available_port(start: int = 7860, stop: int = 7900) -> int:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"Aucun port libre trouve entre {start} et {stop}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lancer l'interface Gradio de conso_predict.")
    parser.add_argument("--share", action="store_true", help="Cree un lien public Gradio.")
    parser.add_argument("--host", help="Adresse d'ecoute locale.")
    parser.add_argument("--port", type=int, help="Port local Gradio. Si absent, un port libre est choisi.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    on_spaces = running_on_spaces()
    host = args.host or ("0.0.0.0" if on_spaces else "127.0.0.1")
    port = args.port if args.port is not None else int(os.getenv("PORT", "7860")) if on_spaces else find_available_port()
    share = args.share and not on_spaces
    print(f"Interface disponible sur http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, share=share, debug=not on_spaces)


if __name__ == "__main__":
    main()
