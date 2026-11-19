import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image  # Ajoute en haut du fichier

# Charger les données_
df_data = pd.read_csv("incidents.csv")
df_casernes = pd.read_csv("casernes.csv")  # Casernes avec colonnes : id, nom, lat, lon, niveau_qualification

# Stockage des incidents filtrés
df_filtre = pd.DataFrame()

def rechercher_incidents(type_incident, mois):
    global df_filtre

    if not mois:
        return None, None, "Veuillez sélectionner un mois", gr.update(visible=False)

    # Filtrer incidents
    df_filtre = df_data[df_data["mois"] == mois]
    if type_incident != "Tous":
        df_filtre = df_filtre[df_filtre["type"] == type_incident]

    if df_filtre.empty:
        return None, None, "Aucun incident trouvé", gr.update(visible=False)

    # Carte incidents
    fig_incidents = px.scatter_mapbox(
        df_filtre,
        lat="lat", lon="lon",
        hover_name="nom",
        hover_data={"type": True, "id": True, "lat": False, "lon": False},
        color="type",
        zoom=8,
        mapbox_style="open-street-map",
        custom_data=["id"]
    )
    fig_incidents.update_traces(marker=dict(size=12))

    # Carte casernes
    fig_casernes = px.scatter_mapbox(
        df_casernes,
        lat="lat", lon="lon",
        hover_name="nom",
        color="niveau_qualification",  # Coloration par niveau
        zoom=8,
        mapbox_style="open-street-map"
    )
    fig_casernes.update_traces(marker=dict(size=30, opacity=0.7))

    fig_incidents.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
    fig_casernes.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})

    return fig_incidents, fig_casernes, f"{len(df_filtre)} incidents affichés", gr.update(choices=df_filtre["id"].astype(str).tolist(), visible=True)

def afficher_stats(incident_id):
    global df_filtre
    if df_filtre.empty or not incident_id:
        return "Aucune donnée disponible.", None

    incident = df_filtre[df_filtre["id"] == int(incident_id)].iloc[0]

    description = f"""
### Incident : {incident['nom']} ({incident['type']})
- **Pompiers sous-qualifiés** : {incident['pompiers_sous_q']}
- **Véhicules sous-qualifiés** : {incident['vehicules_sous_q']}
- **Temps d'intervention** : {incident['temps']} minutes
- **Distance moyenne** : {incident['distance']} km
"""

    # Graphique matplotlib
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ['Pompiers', 'Véhicules', 'Temps (min)', 'Distance (km)']
    values = [
        incident['pompiers_sous_q'],
        incident['vehicules_sous_q'],
        incident['temps'],
        incident['distance']
    ]
    ax.bar(bars, values, color=['red', 'orange', 'blue', 'green'])
    ax.set_title("Statistiques d'intervention")
    plt.tight_layout()

    # Convertir matplotlib -> image PIL
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)

    return description, image

# === INTERFACE GRADIO ===
with gr.Blocks(title="Carte interactive des incidents - Haute-Garonne") as app:
    gr.Markdown("## 🗺️ Visualisation des incidents en Haute-Garonne")

    with gr.Row():
        type_input = gr.Dropdown(["Tous", "Incendie", "Accident", "Inondation", "Secours", "Fuite de gaz"], label="Type d'incident")
        mois_input = gr.Dropdown(["Janvier", "Février", "Mars", "Avril", "Mai", "Juin"], label="Mois (obligatoire)")
        bouton = gr.Button("Démarrer")

    gr.Markdown("### 📍 Carte des incidents")
    sortie_carte = gr.Plot(label="Carte des incidents")

    gr.Markdown("### 🚒 Carte des casernes qualifiées")
    sortie_casernes = gr.Plot(label="Carte des casernes")

    message = gr.Markdown()
    dropdown_incident = gr.Dropdown(label="Cliquez sur un point ou choisissez un ID", visible=False)

    with gr.Row():
        stats_output = gr.Markdown()
        img_plot = gr.Image(type="pil")

    bouton.click(
        fn=rechercher_incidents,
        inputs=[type_input, mois_input],
        outputs=[sortie_carte, sortie_casernes, message, dropdown_incident]
    )

    dropdown_incident.change(
        fn=afficher_stats,
        inputs=dropdown_incident,
        outputs=[stats_output, img_plot]
    )

app.launch(debug=True)
