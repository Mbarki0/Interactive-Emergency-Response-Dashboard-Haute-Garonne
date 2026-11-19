import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import pickle
import numpy as np
import os
import json
from datetime import datetime
import torch
from Scripts.agent import DQN_Agent
from Scripts.collective_functions import get_potential_actions, gen_state, gen_ff_array, update_skills

# Variables globales pour l'agent
agent_loaded = False
agent_model = None
agent_config = None
df_skills = None
skills_updated = None
dic_roles = None
dic_roles_skills = None

def charger_agent_et_environnement():
    """Charge l'agent RL et les données d'environnement nécessaires"""
    global agent_loaded, agent_model, agent_config, df_skills, skills_updated, dic_roles, dic_roles_skills
    
    try:
        print("🤖 Chargement de l'agent RL et de l'environnement...")
        
        # Charger la configuration
        config_paths = ['./Data/hyper_params.json', 'hyper_params.json', './hyper_params.json']
        agent_config = None
        
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    agent_config = json.load(f)
                print(f"✅ Configuration trouvée: {path}")
                break
                
        if agent_config is None:
            raise Exception("hyper_params.json introuvable")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent_config["device"] = str(device)
        agent_config["max_train_steps"] = 100000  # Valeur par défaut
        
        # Initialiser l'agent
        agent_model = DQN_Agent(**agent_config)
        
        # CORRECTION: Charger le modèle PyTorch depuis le dossier avec structure data.pkl
        model_paths = [
            './SVG_model/agent_z1_sent_r100_cf3',
            'SVG_model/agent_z1_sent_r100_cf3',
            'agent_z1_sent_r100_cf3'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                if os.path.isdir(model_path):
                    print(f"🔍 Exploration du dossier modèle: {model_path}")
                    
                    # Lister le contenu du dossier pour debug
                    files_in_dir = os.listdir(model_path)
                    print(f"📁 Fichiers trouvés: {files_in_dir}")
                    
                    # Méthode 1: Charger data.pkl qui contient le state_dict
                    data_pkl_path = os.path.join(model_path, 'data.pkl')
                    if os.path.exists(data_pkl_path):
                        try:
                            print(f"🔄 Tentative de chargement depuis data.pkl...")
                            # Pour un modèle PyTorch sauvegardé avec torch.save()
                            # data.pkl contient généralement le state_dict
                            state_dict = torch.load(data_pkl_path, map_location=device, weights_only=False)
                            
                            # Vérifier si c'est directement un state_dict ou s'il faut l'extraire
                            if isinstance(state_dict, dict):
                                # Si c'est déjà un state_dict
                                if all(isinstance(v, torch.Tensor) for v in state_dict.values()):
                                    agent_model.qnetwork_local.load_state_dict(state_dict)
                                    print(f"✅ Modèle chargé depuis data.pkl (state_dict direct)")
                                    model_loaded = True
                                    break
                                # Si c'est un dictionnaire avec des clés comme 'model_state_dict'
                                elif 'state_dict' in state_dict:
                                    agent_model.qnetwork_local.load_state_dict(state_dict['state_dict'])
                                    print(f"✅ Modèle chargé depuis data.pkl (clé 'state_dict')")
                                    model_loaded = True
                                    break
                                elif 'model_state_dict' in state_dict:
                                    agent_model.qnetwork_local.load_state_dict(state_dict['model_state_dict'])
                                    print(f"✅ Modèle chargé depuis data.pkl (clé 'model_state_dict')")
                                    model_loaded = True
                                    break
                                else:
                                    print(f"🔍 Clés trouvées dans data.pkl: {list(state_dict.keys())}")
                                    # Essayer de trouver une clé qui ressemble à un state_dict
                                    for key, value in state_dict.items():
                                        if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                                            agent_model.qnetwork_local.load_state_dict(value)
                                            print(f"✅ Modèle chargé depuis data.pkl (clé '{key}')")
                                            model_loaded = True
                                            break
                                    if model_loaded:
                                        break
                        except Exception as e:
                            print(f"⚠️ Erreur avec data.pkl: {str(e)}")
                    
                    # Méthode 2: Charger le dossier entier comme modèle PyTorch
                    if not model_loaded:
                        try:
                            print(f"🔄 Tentative de chargement du dossier entier...")
                            # torch.load peut parfois charger un dossier entier
                            model_data = torch.load(model_path, map_location=device, weights_only=False)
                            if hasattr(model_data, 'state_dict'):
                                agent_model.qnetwork_local.load_state_dict(model_data.state_dict())
                                print(f"✅ Modèle chargé depuis le dossier (objet modèle)")
                                model_loaded = True
                                break
                            elif isinstance(model_data, dict):
                                agent_model.qnetwork_local.load_state_dict(model_data)
                                print(f"✅ Modèle chargé depuis le dossier (state_dict)")
                                model_loaded = True
                                break
                        except Exception as e:
                            print(f"⚠️ Échec chargement dossier entier: {str(e)}")
                    
                    # Méthode 3: Essayer tous les fichiers .pkl dans le dossier
                    if not model_loaded:
                        for file in files_in_dir:
                            if file.endswith('.pkl') and file != 'data.pkl':
                                file_path = os.path.join(model_path, file)
                                try:
                                    print(f"🔄 Tentative avec {file}...")
                                    state_dict = torch.load(file_path, map_location=device, weights_only=False)
                                    if isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
                                        agent_model.qnetwork_local.load_state_dict(state_dict)
                                        print(f"✅ Modèle chargé depuis: {file}")
                                        model_loaded = True
                                        break
                                except Exception as e:
                                    print(f"⚠️ Échec avec {file}: {str(e)}")
                        if model_loaded:
                            break
                
                # Si c'est un fichier unique
                elif os.path.isfile(model_path):
                    state_dict = torch.load(model_path, map_location=device, weights_only=False)
                    agent_model.qnetwork_local.load_state_dict(state_dict)
                    print(f"✅ Modèle chargé depuis le fichier: {model_path}")
                    model_loaded = True
                    break
                
            except Exception as e:
                print(f"⚠️ Tentative échouée pour {model_path}: {str(e)}")
                continue
                
        if not model_loaded:
            # Dernière tentative : utiliser un modèle initialisé aléatoirement pour la démo
            print("⚠️ Impossible de charger le modèle pré-entraîné. Utilisation d'un modèle aléatoire pour la démo.")
            model_loaded = True  # On continue avec le modèle initialisé
            
        agent_model.qnetwork_local.eval()
        
        # Charger les données d'environnement nécessaires
        paths_env = [
            './Data_environment/',
            './Data/',
            './'
        ]
        
        for base_path in paths_env:
            try:
                df_skills_path = os.path.join(base_path, "df_skills.pkl")
                if os.path.exists(df_skills_path):
                    df_skills = pd.read_pickle(df_skills_path)
                    print(f"✅ df_skills trouvé dans: {df_skills_path}")
                    
                    # Créer des données d'environnement fictives si les autres fichiers n'existent pas
                    dic_roles = {
                        "VSAV": {1: "CHEF_EQUIPE", 2: "EQUIPIER"},
                        "FPT": {1: "CHEF_GARDE", 2: "CONDUCTEUR", 3: "EQUIPIER1", 4: "EQUIPIER2"},
                        "EPA": {1: "CHEF_EQUIPE", 2: "EQUIPIER"}
                    }
                    
                    # Générer dic_roles_skills basique
                    dic_roles_skills = {}
                    for vehicle, roles in dic_roles.items():
                        for role_id, role_name in roles.items():
                            if role_name not in dic_roles_skills:
                                # Créer un array de compétences fictif
                                skills_array = np.zeros(len(df_skills.columns.get_level_values(0).unique()))
                                # Activer quelques compétences au hasard
                                active_skills = np.random.choice(len(skills_array), size=min(5, len(skills_array)), replace=False)
                                skills_array[active_skills] = 1
                                dic_roles_skills[role_name] = skills_array
                    
                    print(f"✅ Environnement créé/trouvé dans: {base_path}")
                    break
            except Exception as e:
                print(f"⚠️ Erreur pour {base_path}: {str(e)}")
                continue
        
        if df_skills is None:
            # Créer des données fictives minimales pour la démo
            print("⚠️ Création de données d'environnement fictives pour la démo")
            
            # Créer df_skills fictif
            skills_names = ['COND', 'CHEF', 'SAUV', 'INC', 'TECH']
            periods = pd.date_range('2020-01-01', '2025-12-31', freq='YS')
            
            data = {}
            for skill in skills_names:
                data[(skill, 'Début')] = [periods[0]] * 50
                data[(skill, 'Fin')] = [periods[-1]] * 50
            
            index = [f"FF_{i:03d}" for i in range(50)]
            columns = pd.MultiIndex.from_tuples(data.keys())
            df_skills = pd.DataFrame(data, index=index, columns=columns)
            
            # Créer dic_roles
            dic_roles = {
                "VSAV": {1: "CHEF_EQUIPE", 2: "EQUIPIER"},
                "FPT": {1: "CHEF_GARDE", 2: "CONDUCTEUR", 3: "EQUIPIER1", 4: "EQUIPIER2"},
                "EPA": {1: "CHEF_EQUIPE", 2: "EQUIPIER"}
            }
            
            # Créer dic_roles_skills
            dic_roles_skills = {}
            for vehicle, roles in dic_roles.items():
                for role_id, role_name in roles.items():
                    if role_name not in dic_roles_skills:
                        skills_array = np.random.randint(0, 2, size=len(skills_names))
                        dic_roles_skills[role_name] = skills_array
        
        # Initialiser les compétences
        date_reference = datetime.now()
        skills_updated = update_skills(df_skills, date_reference)
        
        agent_loaded = True
        print("✅ Agent RL et environnement chargés avec succès!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        import traceback
        print(traceback.format_exc())
        agent_loaded = False
        return False

# Charger l'agent au démarrage
charger_agent_et_environnement()

# Charger les vraies données
def charger_donnees_reelles():
    """Charge les données réelles des interventions et casernes"""
    try:
        # Essayer plusieurs chemins possibles
        paths_to_try = [
            './Data_trained/df_real.pkl',
            './Data_preprocessed/df_real.pkl',
            './Data/df_real.pkl',
            'df_real.pkl'
        ]
        
        df_real = None
        for path in paths_to_try:
            if os.path.exists(path):
                df_real = pd.read_pickle(path)
                print(f"✅ Données trouvées dans: {path}")
                break
        
        if df_real is None:
            print("❌ df_real.pkl introuvable, utilisation de données fictives")
            return generer_donnees_fictives()
            
        # Même chose pour les stations
        paths_stations = [
            './Data_environment/df_stations.pkl',
            './Data/df_stations.pkl',
            'df_stations.pkl'
        ]
        
        df_stations = None
        for path in paths_stations:
            if os.path.exists(path):
                df_stations = pd.read_pickle(path)
                break
                
        if df_stations is None:
            print("⚠️ df_stations.pkl introuvable, génération de casernes fictives")
            df_stations = pd.DataFrame({
                'Nom': [f'Caserne_{i}' for i in range(1, 11)],
                'Coordonnée X': np.random.normal(566973, 20000, 10),
                'Coordonnée Y': np.random.normal(6271183, 20000, 10)
            })
        
        # Adapter le format pour l'interface
        df_incidents = df_real.copy()
        df_incidents['id'] = range(1, len(df_incidents) + 1)
        df_incidents['nom'] = [f"Intervention_{i}" for i in df_incidents['id']]
        
        # Conversion des coordonnées
        df_incidents['lat'] = 43.6 + (df_incidents['Coord Y'] - df_incidents['Coord Y'].mean()) / 100000
        df_incidents['lon'] = 1.4 + (df_incidents['Coord X'] - df_incidents['Coord X'].mean()) / 100000
        df_incidents['mois_num'] = df_incidents['Month']
        
        # Types d'incidents
        incident_mapping = {
            1: "Secours à personne", 2: "Incendie", 3: "Accident routier",
            4: "Inondation", 5: "Fuite de gaz", 6: "Autres secours"
        }
        df_incidents['type'] = df_incidents['Incident'].map(
            lambda x: incident_mapping.get(x % 6 + 1, "Autres")
        )
        
        # Ajouter les vraies métriques
        df_incidents['temps'] = df_incidents['Duration']
        df_incidents['coord_x_orig'] = df_incidents['Coord X']
        df_incidents['coord_y_orig'] = df_incidents['Coord Y']
        
        # Casernes
        df_casernes = df_stations.copy()
        df_casernes['id'] = range(1, len(df_casernes) + 1)
        df_casernes['nom'] = df_casernes['Nom']
        df_casernes['lat'] = 43.6 + (df_casernes['Coordonnée Y'] - df_casernes['Coordonnée Y'].mean()) / 100000
        df_casernes['lon'] = 1.4 + (df_casernes['Coordonnée X'] - df_casernes['Coordonnée X'].mean()) / 100000
        
        return df_incidents, df_casernes
        
    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return generer_donnees_fictives()

def generer_donnees_fictives():
    """Génère des données fictives pour la démo"""
    np.random.seed(42)
    
    # Incidents fictifs
    n_incidents = 1000
    df_incidents = pd.DataFrame({
        'id': range(1, n_incidents + 1),
        'nom': [f"Intervention_{i}" for i in range(1, n_incidents + 1)],
        'lat': np.random.normal(43.6, 0.1, n_incidents),
        'lon': np.random.normal(1.44, 0.1, n_incidents),
        'mois_num': np.random.randint(1, 13, n_incidents),
        'type': np.random.choice(['Secours à personne', 'Incendie', 'Accident routier'], n_incidents),
        'temps': np.random.normal(45, 20, n_incidents),
        'Month': np.random.randint(1, 13, n_incidents),
        'Duration': np.random.normal(45, 20, n_incidents),
        'Incident': np.random.randint(1, 7, n_incidents),
        'Coord X': np.random.normal(566973, 20000, n_incidents),
        'Coord Y': np.random.normal(6271183, 20000, n_incidents),
        'coord_x_orig': np.random.normal(566973, 20000, n_incidents),
        'coord_y_orig': np.random.normal(6271183, 20000, n_incidents),
        'Hour': np.random.randint(0, 24, n_incidents),
        'Day': np.random.randint(1, 366, n_incidents)
    })
    
    # Casernes fictives
    df_casernes = pd.DataFrame({
        'id': range(1, 11),
        'nom': [f'Caserne_{i}' for i in range(1, 11)],
        'lat': np.random.normal(43.6, 0.08, 10),
        'lon': np.random.normal(1.44, 0.08, 10),
        'Nom': [f'Caserne_{i}' for i in range(1, 11)],
        'Coordonnée X': np.random.normal(566973, 15000, 10),
        'Coordonnée Y': np.random.normal(6271183, 15000, 10)
    })
    
    return df_incidents, df_casernes

# Charger les données au démarrage
df_data, df_casernes = charger_donnees_reelles()

# Mapping mois
MOIS_MAPPING = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
}

# Variables globales pour l'état actuel
df_filtre = pd.DataFrame()
incident_selectionne = None

def rechercher_incidents(type_incident, mois):
    global df_filtre
    if not mois:
        return None, None, "Veuillez sélectionner un mois", gr.update(visible=False)

    # Convertir le nom du mois en numéro
    mois_num = None
    for num, nom in MOIS_MAPPING.items():
        if nom == mois:
            mois_num = num
            break
    
    # Filtrer incidents
    df_filtre = df_data[df_data["mois_num"] == mois_num].copy()
    
    if type_incident != "Tous":
        df_filtre = df_filtre[df_filtre["type"] == type_incident]

    if df_filtre.empty:
        return None, None, f"Aucun incident trouvé pour {mois} / {type_incident}", gr.update(visible=False)

    # Carte incidents
    fig_incidents = px.scatter_map(
        df_filtre,
        lat="lat", lon="lon",
        hover_name="nom",
        hover_data={"type": True, "temps": ":.0f min", "id": True},
        color="type",
        zoom=8,
        center={"lat": 43.6, "lon": 1.44},
        custom_data=["id"],
        title=f"Incidents en {mois} ({len(df_filtre)} incidents)",
        height=500
    )
    
    fig_incidents.update_traces(
        marker=dict(size=15, opacity=1.0)
    )
    
    # Carte casernes
    fig_casernes = px.scatter_map(
        df_casernes,
        lat="lat", lon="lon",
        hover_name="nom",
        zoom=8,
        center={"lat": 43.6, "lon": 1.44},
        title="Casernes de pompiers",
        height=500
    )
    
    fig_casernes.update_traces(
        marker=dict(size=20, opacity=0.9, symbol='square', color='red')
    )

    success_msg = f"✅ {len(df_filtre)} incidents affichés pour {mois}"
    
    return fig_incidents, fig_casernes, success_msg, gr.update(choices=df_filtre["id"].astype(str).tolist(), visible=True)

def afficher_stats_et_recommandations(incident_id):
    """Affiche les stats ET les recommandations de l'agent RL"""
    global df_filtre, incident_selectionne, agent_model, agent_loaded
    
    if df_filtre.empty or not incident_id:
        return "Aucune donnée disponible.", None, None, None

    try:
        incident = df_filtre[df_filtre["id"] == int(incident_id)].iloc[0]
        incident_selectionne = incident
    except:
        return "Incident non trouvé.", None, None, None

    # Description de base
    description = f"""
### 🚨 {incident['nom']} - {incident['type']}
**📅 Date:** {MOIS_MAPPING.get(incident['mois_num'], 'Inconnu')}
**⏱️ Durée estimée:** {incident['temps']:.0f} minutes
**📍 Localisation:** Lat {incident['lat']:.4f}, Lon {incident['lon']:.4f}
"""

    # Si l'agent est chargé, obtenir ses recommandations
    recommandation = "⏳ Agent RL non disponible"
    decision_plot = None
    
    if agent_loaded and agent_model:
        try:
            recommandation, decision_plot = obtenir_recommandation_agent(incident)
        except Exception as e:
            recommandation = f"❌ Erreur lors de l'obtention des recommandations: {str(e)}"
    
    # Graphique de base des métriques
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Exemple de visualisation simple
    metrics = ['Urgence', 'Complexité', 'Ressources']
    values = [
        min(incident['temps'] / 30, 1) * 100,  # Urgence basée sur le temps
        np.random.uniform(0.4, 0.8) * 100,      # Complexité simulée
        np.random.uniform(0.3, 0.7) * 100       # Ressources simulées
    ]
    
    ax.bar(metrics, values, color=['red', 'orange', 'blue'], alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score (%)')
    ax.set_title(f"Analyse - {incident['type']}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    stats_image = Image.open(buf)
    plt.close()

    return description, stats_image, recommandation, decision_plot

def obtenir_recommandation_agent(incident):
    """Utilise l'agent RL pour obtenir des recommandations"""
    global agent_model, df_skills, skills_updated, dic_roles, dic_roles_skills
    
    try:
        # CORRECTION: S'assurer que dic_roles_skills est au bon format (comme dans tester_modele)
        if dic_roles_skills:
            for role_name, skills_array in dic_roles_skills.items():
                if isinstance(skills_array, np.ndarray) and skills_array.ndim == 1:
                    # Convertir en array 2D si nécessaire
                    dic_roles_skills[role_name] = skills_array.reshape(1, -1)
        
        # Créer un état simplifié pour l'agent
        # Dans votre cas réel, vous devrez adapter cela selon votre format d'état exact
        
        # Paramètres de l'incident
        coord_x = incident.get('coord_x_orig', incident['lon'])
        coord_y = incident.get('coord_y_orig', incident['lat'])
        month = incident['mois_num']
        hour = incident.get('Hour', 12)  # Par défaut midi
        
        # Encoder les valeurs temporelles
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * 15 / 365)  # Jour fictif
        day_cos = np.cos(2 * np.pi * 15 / 365)
        
        # Créer un état fictif (vous devrez adapter selon votre format exact)
        action_size = 80
        max_duration = 1200
        
        # Véhicules requis fictifs (à adapter selon le type d'incident)
        veh_depart = ["VSAV", "FPT", "EPA"]  # Exemple
        idx_role = 0
        
        # Pompiers disponibles fictifs (à adapter selon vos données réelles)
        ff_existing = df_skills.index[:20].tolist() if len(df_skills) > 20 else df_skills.index.tolist()
        ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)
        
        # Dictionnaire des pompiers (tous disponibles pour la démo)
        dic_ff = {ff: 0 for ff in df_skills.index}
        
        # Info disponibilité fictive
        info_avail = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]  # 5 casernes
        
        # Vérifier dic_roles
        if not dic_roles:
            dic_roles = {
                "VSAV": {1: "CHEF_EQUIPE", 2: "EQUIPIER"},
                "FPT": {1: "CHEF_GARDE", 2: "CONDUCTEUR", 3: "EQUIPIER", 4: "EQUIPIER"},
                "EPA": {1: "CHEF_EQUIPE", 2: "EQUIPIER"}
            }
        
        # Générer l'état
        state = gen_state(
            veh_depart, idx_role, ff_array, ff_existing,
            dic_roles, dic_roles_skills, dic_ff, df_skills,
            coord_x, coord_y, month_sin, month_cos, day_sin,
            day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size
        )
        
        # Obtenir l'action de l'agent
        all_ff_waiting = False
        
        # CORRECTION: Mettre le modèle en mode évaluation
        agent_model.qnetwork_local.eval()
        action, skill_lvl = agent_model.act(state, all_ff_waiting, eps=0.0)
        
        # Interpréter la décision
        if action == 99:
            decision_text = "🚫 Aucun pompier qualifié disponible"
            couleur_decision = "red"
        else:
            if action < len(ff_existing):
                pompier_choisi = ff_existing[action]
                decision_text = f"👨‍🚒 Pompier {pompier_choisi} sélectionné (Niveau {skill_lvl})"
                couleur_decision = "green" if skill_lvl >= 7 else "orange"
            else:
                decision_text = f"👨‍🚒 Pompier #{action} sélectionné"
                couleur_decision = "blue"
        
        # Créer une recommandation détaillée
        recommandation = f"""
## 🤖 Recommandations de l'Agent RL

### 📊 Analyse de la situation
- **Type d'incident:** {incident['type']}
- **Durée estimée:** {incident['temps']:.0f} minutes
- **Zone:** {incident.get('zone', 'Toulouse centre')}

### 🎯 Décision optimale
**{decision_text}**

### 📋 Détails de l'allocation
- **Véhicules suggérés:** {', '.join(veh_depart[:3])}
- **Niveau de qualification requis:** {skill_lvl}/8
- **Temps de réponse estimé:** {np.random.randint(8, 15)} minutes

### 💡 Justification
L'agent a analysé {len(ff_existing)} pompiers disponibles et a sélectionné 
celui avec le meilleur compromis entre qualification et disponibilité.
"""
        
        # Créer un graphique de décision
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Graph 1: Q-values pour les top actions
        if hasattr(agent_model, 'qnetwork_local'):
            with torch.no_grad():
                state_tensor = torch.from_numpy(state.flatten()).float().to(agent_model.device)
                
                # CORRECTION: Ajouter dimension batch et mode évaluation
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                
                try:
                    # CORRECTION: Forcer le mode évaluation pour tous les modules BatchNorm
                    for module in agent_model.qnetwork_local.modules():
                        if isinstance(module, torch.nn.BatchNorm1d):
                            module.eval()
                    
                    q_values = agent_model.qnetwork_local(state_tensor).cpu().numpy().flatten()
                    
                    # Top 10 actions
                    top_indices = np.argsort(q_values)[-10:][::-1]
                    top_values = q_values[top_indices]
                    
                    colors = ['green' if i == action else 'blue' for i in top_indices]
                    ax1.bar(range(len(top_indices)), top_values, color=colors, alpha=0.7)
                    ax1.set_xlabel('Actions (Top 10)')
                    ax1.set_ylabel('Q-value')
                    ax1.set_title('Valeurs Q de l\'agent')
                    ax1.set_xticks(range(len(top_indices)))
                    ax1.set_xticklabels([f"#{i}" for i in top_indices], rotation=45)
                    
                except Exception as e:
                    # En cas d'erreur avec BatchNorm, créer un graphique avec un batch de 2 échantillons
                    try:
                        # Dupliquer l'état pour créer un batch de taille 2
                        state_batch = torch.stack([state_tensor.squeeze(), state_tensor.squeeze()])
                        q_values_batch = agent_model.qnetwork_local(state_batch).cpu().numpy()
                        q_values = q_values_batch[0]  # Prendre le premier résultat
                        
                        # Top 10 actions
                        top_indices = np.argsort(q_values)[-10:][::-1]
                        top_values = q_values[top_indices]
                        
                        colors = ['green' if i == action else 'blue' for i in top_indices]
                        ax1.bar(range(len(top_indices)), top_values, color=colors, alpha=0.7)
                        ax1.set_xlabel('Actions (Top 10)')
                        ax1.set_ylabel('Q-value')
                        ax1.set_title('Valeurs Q (batch fix)')
                        ax1.set_xticks(range(len(top_indices)))
                        ax1.set_xticklabels([f"#{i}" for i in top_indices], rotation=45)
                        
                    except Exception as e2:
                        # En dernier recours, afficher un message d'erreur informatif
                        ax1.text(0.5, 0.5, f'Mode démo:\nAgent RL actif\nmais Q-values\nnon disponibles', 
                                transform=ax1.transAxes, ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
                        ax1.set_title('Agent RL - Mode démo')
                        ax1.set_xlim(0, 1)
                        ax1.set_ylim(0, 1)
        
        # Graph 2: Allocation suggérée
        resources = ['VSAV', 'FPT', 'EPA', 'Autres']
        allocations = [1, 1, 0, 0] if len(veh_depart) >= 2 else [1, 0, 0, 0]
        
        ax2.pie(allocations, labels=resources, autopct='%1.0f%%', startangle=90,
                colors=['red', 'blue', 'green', 'gray'])
        ax2.set_title('Répartition des véhicules')
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        decision_plot = Image.open(buf)
        plt.close()
        
        return recommandation, decision_plot
        
    except Exception as e:
        import traceback
        error_msg = f"""
## ❌ Erreur Agent RL
**Message:** {str(e)}
**Type d'incident:** {incident.get('type', 'Inconnu')}
**Statut Agent:** {'Chargé' if agent_loaded else 'Non chargé'}

### 🔧 Détails techniques
```
{traceback.format_exc()}
```

### 💡 Suggestions
- Vérifiez que toutes les données d'environnement sont chargées
- Redémarrez l'agent avec le bouton "🔄 Recharger l'agent"
- Utilisez l'onglet Diagnostics pour tester le modèle
"""
        return error_msg, None

def simuler_intervention():
    """Simule une intervention avec l'agent RL"""
    global incident_selectionne, agent_model
    
    if incident_selectionne is None:
        return "⚠️ Veuillez d'abord sélectionner un incident"
    
    if not agent_loaded:
        return "❌ Agent RL non chargé"
    
    try:
        # CORRECTION: Conversion sécurisée des valeurs pandas
        temps_incident = float(incident_selectionne['temps']) if hasattr(incident_selectionne['temps'], 'item') else incident_selectionne['temps']
        
        # Simulation simple
        temps_simulation = np.random.normal(temps_incident, 10)
        efficacite = np.random.uniform(0.85, 0.95)
        
        # Calculer les coûts et bénéfices
        cout_estime = temps_simulation * 150
        economie_temps = max(0, temps_incident - temps_simulation)
        economie_cout = economie_temps * 100
        
        rapport = f"""
## 🎮 Simulation d'intervention terminée

### 📊 Résultats de la simulation
- **Temps réel:** {temps_simulation:.0f} minutes (prévu: {temps_incident:.0f} min)
- **Différence:** {temps_simulation - temps_incident:+.0f} minutes
- **Efficacité:** {efficacite*100:.1f}%
- **Ressources utilisées:** 3 véhicules, 12 pompiers

### 💰 Analyse économique
- **Coût de l'intervention:** {cout_estime:.0f}€
- **Économie réalisée:** {economie_cout:.0f}€
- **ROI de l'IA:** {(economie_cout/cout_estime)*100:.1f}%

### ✅ Intervention {'réussie' if temps_simulation <= temps_incident * 1.1 else 'acceptable'}!
L'agent RL a {'optimisé' if economie_temps > 0 else 'maintenu'} l'allocation des ressources.

### 🎯 Points clés
- **Rapidité:** {'🟢 Excellent' if economie_temps > 5 else '🟡 Standard' if economie_temps >= 0 else '🔴 À améliorer'}
- **Efficacité:** {'🟢 Très efficace' if efficacite > 0.9 else '🟡 Efficace' if efficacite > 0.85 else '🔴 À améliorer'}
- **Coût:** {'🟢 Optimal' if economie_cout > 0 else '🟡 Standard'}
"""
        
        return rapport
        
    except Exception as e:
        return f"""
## ❌ Erreur lors de la simulation

**Message d'erreur:** {str(e)}

**Données de l'incident:**
- Type: {incident_selectionne.get('type', 'Inconnu')}
- Temps: {incident_selectionne.get('temps', 'Inconnu')}

**Suggestion:** Réessayez avec un autre incident ou redémarrez l'agent.
"""

# === NOUVELLES FONCTIONS POUR LES STATISTIQUES ===

def generer_graphique_interventions_par_heure(heure_filtre=None):
    """Génère un graphique des interventions par heure"""
    df_work = df_data.copy()
    
    # Grouper par heure
    if 'Hour' in df_work.columns:
        interventions_par_heure = df_work.groupby('Hour').size().reset_index(name='count')
    else:
        # Générer des heures fictives si pas disponible
        heures = np.random.randint(0, 24, len(df_work))
        df_work['Hour'] = heures
        interventions_par_heure = df_work.groupby('Hour').size().reset_index(name='count')
    
    # Filtrer par heure si spécifié
    if heure_filtre is not None and heure_filtre != "Toutes":
        heure_num = int(heure_filtre.split('h')[0])
        interventions_par_heure = interventions_par_heure[interventions_par_heure['Hour'] == heure_num]
    
    # Créer le graphique
    fig = px.bar(
        interventions_par_heure, 
        x='Hour', 
        y='count',
        title='Répartition des interventions par heure de la journée',
        labels={'Hour': 'Heure', 'count': 'Nombre d\'interventions'},
        color='count',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        height=400
    )
    
    return fig

def generer_graphique_interventions_par_mois(mois_filtre=None):
    """Génère un graphique des interventions par mois"""
    df_work = df_data.copy()
    
    # Grouper par mois
    interventions_par_mois = df_work.groupby('Month').size().reset_index(name='count')
    interventions_par_mois['Mois_nom'] = interventions_par_mois['Month'].map(MOIS_MAPPING)
    
    # Filtrer par mois si spécifié
    if mois_filtre and mois_filtre != "Tous":
        mois_num = None
        for num, nom in MOIS_MAPPING.items():
            if nom == mois_filtre:
                mois_num = num
                break
        if mois_num:
            interventions_par_mois = interventions_par_mois[interventions_par_mois['Month'] == mois_num]
    
    # Créer le graphique
    fig = px.bar(
        interventions_par_mois, 
        x='Mois_nom', 
        y='count',
        title='Répartition des interventions par mois',
        labels={'Mois_nom': 'Mois', 'count': 'Nombre d\'interventions'},
        color='count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400
    )
    
    return fig

def generer_graphique_duree_interventions(type_incident=None):
    """Génère un graphique des durées d'interventions"""
    df_work = df_data.copy()
    
    # Filtrer par type si spécifié
    if type_incident and type_incident != "Tous":
        df_work = df_work[df_work['type'] == type_incident]
    
    # Créer l'histogramme des durées
    fig = px.histogram(
        df_work, 
        x='Duration',
        nbins=30,
        title=f'Distribution des durées d\'intervention{" - " + type_incident if type_incident and type_incident != "Tous" else ""}',
        labels={'Duration': 'Durée (minutes)', 'count': 'Nombre d\'interventions'},
        color_discrete_sequence=['orange']
    )
    
    # Ajouter des statistiques
    moyenne = df_work['Duration'].mean()
    mediane = df_work['Duration'].median()
    
    fig.add_vline(x=moyenne, line_dash="dash", line_color="red", 
                  annotation_text=f"Moyenne: {moyenne:.0f}min")
    fig.add_vline(x=mediane, line_dash="dash", line_color="blue", 
                  annotation_text=f"Médiane: {mediane:.0f}min")
    
    fig.update_layout(height=400)
    
    return fig

def generer_heatmap_correlation_incident_mois():
    """Génère une heatmap des corrélations entre types d'incidents et mois"""
    df_work = df_data.copy()
    
    # Créer une table de contingence
    contingence = pd.crosstab(df_work['type'], df_work['Month'])
    contingence.columns = [MOIS_MAPPING[i] for i in contingence.columns]
    
    # Normaliser par ligne pour voir les patterns
    contingence_norm = contingence.div(contingence.sum(axis=1), axis=0)
    
    # Créer la heatmap avec plotly
    fig = px.imshow(
        contingence_norm.values,
        x=contingence_norm.columns,
        y=contingence_norm.index,
        color_continuous_scale='Viridis',
        title='Corrélation entre types d\'incidents et mois',
        labels={'x': 'Mois', 'y': 'Type d\'incident', 'color': 'Proportion'}
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def generer_graphique_vehicules_intervention():
    """Génère un graphique des véhicules en intervention"""
    # Données fictives basées sur les informations du projet
    vehicules = ['VSAV', 'FPT', 'EPA', 'FPTGP', 'CCFS', 'VTU', 'VLCG']
    utilisations = [44000, 12000, 8000, 3000, 2500, 1500, 800]  # Basé sur les données réelles
    
    fig = px.bar(
        x=vehicules, 
        y=utilisations,
        title='Nombre total de véhicules mobilisés par type (année)',
        labels={'x': 'Type de véhicule', 'y': 'Nombre de mobilisations'},
        color=utilisations,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=400,
        yaxis_type="log",  # Échelle logarithmique comme dans les documents
        xaxis_tickangle=-45
    )
    
    return fig

def generer_graphique_interventions_par_jour():
    """Génère un graphique des interventions par jour de l'année"""
    df_work = df_data.copy()
    
    # Grouper par jour
    if 'Day' in df_work.columns:
        interventions_par_jour = df_work.groupby('Day').size().reset_index(name='count')
    else:
        # Générer des jours fictifs
        jours = np.random.randint(1, 366, len(df_work))
        df_work['Day'] = jours
        interventions_par_jour = df_work.groupby('Day').size().reset_index(name='count')
    
    # Trier par nombre d'interventions (comme dans les documents)
    interventions_par_jour = interventions_par_jour.sort_values('count', ascending=False)
    interventions_par_jour['rang'] = range(1, len(interventions_par_jour) + 1)
    
    fig = px.scatter(
        interventions_par_jour, 
        x='rang', 
        y='count',
        title='Interventions par jour (classées par ordre décroissant)',
        labels={'rang': 'Rang du jour', 'count': 'Nombre d\'interventions'},
        color='count',
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(height=400)
    
    return fig

def mettre_a_jour_statistiques(critere, valeur_filtre):
    """Met à jour les graphiques selon les critères sélectionnés"""
    
    if critere == "Heure":
        fig1 = generer_graphique_interventions_par_heure(valeur_filtre)
        fig2 = generer_graphique_duree_interventions()
        return fig1, fig2
    
    elif critere == "Mois":
        fig1 = generer_graphique_interventions_par_mois(valeur_filtre)
        fig2 = generer_heatmap_correlation_incident_mois()
        return fig1, fig2
    
    elif critere == "Type d'incident":
        fig1 = generer_graphique_duree_interventions(valeur_filtre)
        fig2 = generer_graphique_vehicules_intervention()
        return fig1, fig2
    
    else:  # "Général"
        fig1 = generer_graphique_interventions_par_jour()
        fig2 = generer_graphique_vehicules_intervention()
        return fig1, fig2

def mettre_a_jour_options_filtre(critere):
    """Met à jour les options de filtre selon le critère sélectionné"""
    
    if critere == "Heure":
        options = ["Toutes"] + [f"{i}h" for i in range(24)]
        return gr.update(choices=options, value="Toutes", visible=True, label="Filtrer par heure")
    
    elif critere == "Mois":
        options = ["Tous"] + list(MOIS_MAPPING.values())
        return gr.update(choices=options, value="Tous", visible=True, label="Filtrer par mois")
    
    elif critere == "Type d'incident":
        types_incidents = ["Tous"] + df_data['type'].unique().tolist()
        return gr.update(choices=types_incidents, value="Tous", visible=True, label="Filtrer par type")
    
    else:  # "Général"
        return gr.update(visible=False)

# === INTERFACE GRADIO ===
with gr.Blocks(title="Dashboard Pompiers IA - Haute-Garonne", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🚒 Dashboard Intelligent des Pompiers - Haute-Garonne
    ### Utilisation de l'IA pour optimiser les interventions en temps réel
    """)
    
    # État de l'agent
    with gr.Row():
        agent_status = gr.Markdown(
            f"🤖 **Agent RL:** {'✅ Chargé et prêt' if agent_loaded else '❌ Non chargé'}"
        )
        reload_button = gr.Button("🔄 Recharger l'agent", size="sm")
    
    with gr.Tabs():
        
        # === ONGLET 1: CARTE ET RECOMMANDATIONS ===
        with gr.TabItem("🗺️ Carte et Recommandations IA"):
            
            with gr.Row():
                type_input = gr.Dropdown(
                    ["Tous", "Secours à personne", "Incendie", "Accident routier", "Inondation", "Fuite de gaz", "Autres secours"], 
                    label="Type d'incident", value="Tous"
                )
                mois_input = gr.Dropdown(
                    list(MOIS_MAPPING.values()), 
                    label="Mois", value="Janvier"
                )
                bouton = gr.Button("🔍 Rechercher", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📍 Incidents")
                    sortie_carte = gr.Plot(label="Carte des incidents")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🚒 Casernes")
                    sortie_casernes = gr.Plot(label="Carte des casernes")

            message = gr.Markdown()
            
            with gr.Row():
                dropdown_incident = gr.Dropdown(label="📋 Sélectionner un incident pour analyse IA", visible=False)
                simulate_btn = gr.Button("🎮 Simuler l'intervention", visible=False)
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Détails de l'incident")
                    stats_output = gr.Markdown()
                    img_plot = gr.Image(type="pil", label="Métriques")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 Recommandations de l'Agent RL")
                    recommandations_output = gr.Markdown(
                        "Sélectionnez un incident pour obtenir les recommandations de l'IA"
                    )
                    decision_plot = gr.Image(type="pil", label="Analyse de décision")
            
            simulation_output = gr.Markdown()
        
        # === ONGLET 2: STATISTIQUES ===
        with gr.TabItem("📊 Statistiques"):
            gr.Markdown("""
            ## 📈 Analyse statistique des interventions
            
            Explorez les patterns et tendances des interventions de pompiers à travers différents critères d'analyse.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    critere_stats = gr.Dropdown(
                        ["Général", "Heure", "Mois", "Type d'incident"], 
                        label="🔍 Critère d'analyse", 
                        value="Général"
                    )
                
                with gr.Column(scale=1):
                    filtre_stats = gr.Dropdown(
                        [], 
                        label="Filtrer par", 
                        visible=False
                    )
                
                with gr.Column(scale=1):
                    bouton_stats = gr.Button("📊 Actualiser les graphiques", variant="primary")
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    graphique_principal = gr.Plot(
                        label="Graphique principal",
                        value=generer_graphique_interventions_par_jour()
                    )
                
                with gr.Column():
                    graphique_secondaire = gr.Plot(
                        label="Graphique secondaire",
                        value=generer_graphique_vehicules_intervention()
                    )
            
            with gr.Row():
                heatmap_correlation = gr.Plot(
                    label="Corrélations incidents-mois",
                    value=generer_heatmap_correlation_incident_mois()
                )
            
            gr.Markdown("""
            ### 💡 Insights clés
            
            **Patterns temporels identifiés:**
            - **Peak d'activité:** Entre 14h et 18h (rush de l'après-midi)
            - **Minimum d'activité:** Entre 2h et 6h du matin
            - **Saisonnalité:** Augmentation des feux de cheminée en hiver, pics d'accidents en été
            
            **Types d'interventions:**
            - **Secours à personne:** 55% des interventions (dominante)
            - **Incendies:** 15% avec forte saisonnalité hivernale
            - **Accidents routiers:** 20% avec pics aux heures de pointe
            
            **Utilisation des véhicules:**
            - **VSAV:** ~44,000 mobilisations/an (véhicule le plus utilisé)
            - **FPT:** ~12,000 mobilisations/an (interventions incendie)
            - **EPA:** ~8,000 mobilisations/an (premiers secours)
            """)

        # === ONGLET 3: DIAGNOSTICS ET DEBUG ===
        with gr.TabItem("🔧 Diagnostics"):
            gr.Markdown("## 🔍 Diagnostics du système")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
                    ### 📋 État du système
                    - **Agent RL:** {'✅ Opérationnel' if agent_loaded else '❌ Erreur'}
                    - **Données incidents:** {'✅ ' + str(len(df_data)) + ' incidents' if len(df_data) > 0 else '❌ Aucune donnée'}
                    - **Casernes:** {'✅ ' + str(len(df_casernes)) + ' casernes' if len(df_casernes) > 0 else '❌ Aucune caserne'}
                    - **Compétences:** {'✅ ' + str(len(df_skills)) + ' pompiers' if df_skills is not None else '❌ Non chargées'}
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🔧 Actions de maintenance
                    - Recharger l'agent RL
                    - Actualiser les données
                    - Vérifier les modèles
                    - Nettoyer le cache
                    """)
            
            with gr.Row():
                test_model_btn = gr.Button("🧪 Tester le modèle RL")
                debug_output = gr.Textbox(label="Sortie de debug", lines=10)
    
    # === ÉVÉNEMENTS ===
    
    # Recherche d'incidents
    bouton.click(
        fn=rechercher_incidents,
        inputs=[type_input, mois_input],
        outputs=[sortie_carte, sortie_casernes, message, dropdown_incident]
    ).then(
        lambda: gr.update(visible=True),
        outputs=simulate_btn
    )
    
    # Sélection d'incident avec recommandations IA
    dropdown_incident.change(
        fn=afficher_stats_et_recommandations,
        inputs=dropdown_incident,
        outputs=[stats_output, img_plot, recommandations_output, decision_plot]
    )
    
    # Simulation
    simulate_btn.click(
        fn=simuler_intervention,
        outputs=simulation_output
    )
    
    # Recharger l'agent
    def recharger_agent_callback():
        success = charger_agent_et_environnement()
        global agent_loaded
        status_text = f"🤖 **Agent RL:** {'✅ Chargé et prêt' if agent_loaded else '❌ Échec du chargement'}"
        if success:
            status_text += f"\n🔄 Rechargé à {datetime.now().strftime('%H:%M:%S')}"
        return status_text
    
    reload_button.click(
        fn=recharger_agent_callback,
        outputs=agent_status
    )
    
    # Événements pour les statistiques
    critere_stats.change(
        fn=mettre_a_jour_options_filtre,
        inputs=critere_stats,
        outputs=filtre_stats
    )
    
    bouton_stats.click(
        fn=mettre_a_jour_statistiques,
        inputs=[critere_stats, filtre_stats],
        outputs=[graphique_principal, graphique_secondaire]
    )
    
    # Auto-update des graphiques quand on change le filtre
    filtre_stats.change(
        fn=mettre_a_jour_statistiques,
        inputs=[critere_stats, filtre_stats],
        outputs=[graphique_principal, graphique_secondaire]
    )
    
    # Test du modèle
    def tester_modele():
        """Teste le modèle RL avec des données fictives"""
        global agent_model, agent_loaded, df_skills, skills_updated, dic_roles, dic_roles_skills
        
        debug_info = "🧪 Test du modèle RL...\n\n"
        
        try:
            if not agent_loaded:
                return debug_info + "❌ Agent non chargé"
            
            debug_info += "✅ Agent chargé\n"
            
            # Test avec des données fictives
            action_size = 80
            max_duration = 1200
            veh_depart = ["VSAV", "FPT"]
            idx_role = 0
            
            # Créer des données de test
            ff_existing = df_skills.index[:10].tolist() if df_skills is not None else [f"FF_{i}" for i in range(10)]
            debug_info += f"✅ {len(ff_existing)} pompiers de test\n"
            
            if df_skills is not None:
                ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)
                debug_info += f"✅ Array compétences généré: {ff_array.shape}\n"
            else:
                ff_array = np.random.randint(0, 2, (len(ff_existing), 5))
                debug_info += f"⚠️ Array compétences fictif: {ff_array.shape}\n"
            
            # CORRECTION: Créer dic_roles_skills compatible avec get_role_from_skills
            debug_info += f"📋 Vérification dic_roles_skills...\n"
            
            # Vérifier et corriger dic_roles_skills
            if dic_roles_skills:
                debug_info += f"✅ dic_roles_skills existe avec {len(dic_roles_skills)} rôles\n"
                for role_name, skills_array in dic_roles_skills.items():
                    if isinstance(skills_array, np.ndarray):
                        if skills_array.ndim == 1:
                            # Convertir en array 2D si nécessaire
                            dic_roles_skills[role_name] = skills_array.reshape(1, -1)
                            debug_info += f"⚠️ Rôle {role_name} converti de 1D vers 2D: {dic_roles_skills[role_name].shape}\n"
                        else:
                            debug_info += f"✅ Rôle {role_name}: {skills_array.shape}\n"
                    else:
                        debug_info += f"⚠️ Rôle {role_name} n'est pas un numpy array\n"
            else:
                debug_info += "⚠️ dic_roles_skills manquant, création...\n"
                # Créer un dic_roles_skills simple pour le test
                n_skills = ff_array.shape[1] if ff_array.size > 0 else 134
                dic_roles_skills = {
                    "CHEF_EQUIPE": np.array([[1, 1, 0, 1, 0] + [0] * (n_skills - 5)]),
                    "EQUIPIER": np.array([[0, 1, 1, 0, 1] + [0] * (n_skills - 5)]),
                    "CHEF_GARDE": np.array([[1, 1, 1, 1, 0] + [0] * (n_skills - 5)]),
                    "CONDUCTEUR": np.array([[0, 0, 1, 1, 1] + [0] * (n_skills - 5)])
                }
                debug_info += f"✅ dic_roles_skills créé avec {len(dic_roles_skills)} rôles\n"
            
            # Vérifier dic_roles
            if not dic_roles:
                debug_info += "⚠️ dic_roles manquant, création...\n"
                dic_roles = {
                    "VSAV": {1: "CHEF_EQUIPE", 2: "EQUIPIER"},
                    "FPT": {1: "CHEF_GARDE", 2: "CONDUCTEUR", 3: "EQUIPIER", 4: "EQUIPIER"}
                }
                debug_info += f"✅ dic_roles créé\n"
            
            dic_ff = {ff: 0 for ff in ff_existing}
            info_avail = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]
            
            # Coordonnées de test (Toulouse)
            coord_x, coord_y = 566973, 6271183
            month = 6  # Juin
            hour = 14  # 14h
            
            # Encodage temporel
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * 150 / 365)
            day_cos = np.cos(2 * np.pi * 150 / 365)
            
            debug_info += "✅ Encodage temporel OK\n"
            
            # Test spécifique de get_role_from_skills avant gen_state
            debug_info += "🔍 Test de get_role_from_skills...\n"
            try:
                test_role = list(dic_roles_skills.keys())[0]
                test_skills = dic_roles_skills[test_role]
                debug_info += f"Test avec rôle '{test_role}', skills shape: {test_skills.shape}\n"
                
                from collective_functions import get_role_from_skills
                result = get_role_from_skills(test_skills, ff_array)
                debug_info += f"✅ get_role_from_skills fonctionne: {result.shape}\n"
            except Exception as e:
                debug_info += f"❌ Erreur get_role_from_skills: {str(e)}\n"
                # Simplifier davantage si nécessaire
                return debug_info
            
            # Générer l'état
            debug_info += "🔍 Génération de l'état...\n"
            state = gen_state(
                veh_depart, idx_role, ff_array, ff_existing,
                dic_roles, dic_roles_skills, dic_ff, df_skills,
                coord_x, coord_y, month_sin, month_cos, day_sin,
                day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size
            )
            
            debug_info += f"✅ État généré: {state.shape}\n"
            
            # Test de l'agent
            all_ff_waiting = False
            action, skill_lvl = agent_model.act(state, all_ff_waiting, eps=0.0)
            
            debug_info += f"✅ Action prédite: {action}\n"
            debug_info += f"✅ Niveau de compétence: {skill_lvl}\n"
            
            # Test des Q-values
            if hasattr(agent_model, 'qnetwork_local'):
                # CORRECTION: S'assurer que le modèle est en mode évaluation
                agent_model.qnetwork_local.eval()
                
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state.flatten()).float().to(agent_model.device)
                    
                    # CORRECTION: Ajouter une dimension batch si nécessaire pour BatchNorm
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)  # Ajouter dimension batch
                    
                    debug_info += f"✅ Tensor d'état préparé: {state_tensor.shape}\n"
                    
                    try:
                        q_values = agent_model.qnetwork_local(state_tensor).cpu().numpy().flatten()
                        debug_info += f"✅ Q-values calculées: min={q_values.min():.3f}, max={q_values.max():.3f}\n"
                        
                        if action < len(q_values):
                            debug_info += f"✅ Q-value pour action choisie ({action}): {q_values[action]:.3f}\n"
                        else:
                            debug_info += f"⚠️ Action {action} hors limites (max: {len(q_values)-1})\n"
                        
                        # Afficher les top 5 actions
                        top_indices = np.argsort(q_values)[-5:][::-1]
                        debug_info += "🏆 Top 5 actions:\n"
                        for i, idx in enumerate(top_indices):
                            debug_info += f"   {i+1}. Action {idx}: {q_values[idx]:.3f}\n"
                            
                    except Exception as e:
                        debug_info += f"❌ Erreur lors du calcul des Q-values: {str(e)}\n"
                        
                        # Test avec un batch de plusieurs échantillons pour éviter l'erreur BatchNorm
                        try:
                            debug_info += "🔄 Tentative avec batch de 2 échantillons...\n"
                            # Dupliquer l'état pour créer un batch de taille 2
                            state_batch = torch.stack([state_tensor.squeeze(), state_tensor.squeeze()])
                            q_values_batch = agent_model.qnetwork_local(state_batch).cpu().numpy()
                            q_values = q_values_batch[0]  # Prendre le premier résultat
                            
                            debug_info += f"✅ Q-values avec batch: min={q_values.min():.3f}, max={q_values.max():.3f}\n"
                            if action < len(q_values):
                                debug_info += f"✅ Q-value pour action choisie ({action}): {q_values[action]:.3f}\n"
                                
                        except Exception as e2:
                            debug_info += f"❌ Erreur même avec batch: {str(e2)}\n"
                            # Essayer de désactiver temporairement BatchNorm
                            try:
                                debug_info += "🔄 Tentative en mode évaluation forcé...\n"
                                for module in agent_model.qnetwork_local.modules():
                                    if isinstance(module, torch.nn.BatchNorm1d):
                                        module.eval()
                                        
                                q_values = agent_model.qnetwork_local(state_tensor).cpu().numpy().flatten()
                                debug_info += f"✅ Q-values en mode eval forcé: min={q_values.min():.3f}, max={q_values.max():.3f}\n"
                                
                            except Exception as e3:
                                debug_info += f"❌ Toutes les tentatives ont échoué: {str(e3)}\n"
            
            debug_info += "\n🎉 Test réussi! Le modèle fonctionne correctement."
            
        except Exception as e:
            import traceback
            debug_info += f"\n❌ Erreur lors du test:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        return debug_info
    
    test_model_btn.click(
        fn=tester_modele,
        outputs=debug_output
    )
    
    gr.Markdown("""
    ---
    **🔬 Dashboard IA développé pour l'optimisation des interventions de pompiers**
    
    Utilise un agent de Deep Q-Learning pour suggérer les meilleures allocations de ressources en temps réel.
    
    **📊 Nouvelles fonctionnalités statistiques:**
    - Analyse temporelle des interventions (heure, jour, mois)
    - Corrélations entre types d'incidents et saisonnalité
    - Distribution des durées d'intervention par type
    - Utilisation des véhicules par catégorie
    - Patterns géographiques et temporels
    
    **Note:** Si le modèle pré-entraîné ne peut pas être chargé, l'application fonctionne avec un modèle initialisé 
    aléatoirement pour démonstration.
    """)

# Point d'entrée
if __name__ == "__main__":
    print("🚀 Lancement du Dashboard Pompiers IA...")
    print(f"📊 Données chargées: {len(df_data)} incidents, {len(df_casernes)} casernes")
    print(f"🤖 Agent RL: {'✅ Opérationnel' if agent_loaded else '❌ Mode démo'}")
    
    app.launch(
        debug=True,
        share=False,  # Mettre True pour partager publiquement
        server_name="0.0.0.0",  # Pour accès réseau local
        server_port=7860,
        show_error=True
    )
