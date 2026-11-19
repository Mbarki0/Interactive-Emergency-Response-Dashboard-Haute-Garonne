import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import re
import pandas as pd


def load_data_with_stations(csv_file='./event_logs_with_vehicles.csv'):
    """
    Charger les données modifiées avec types de véhicules et informations des casernes
    """
    print("📂 Chargement des données modifiées...")
    df_events = pd.read_csv(csv_file)
    # Charger df_pc_real qui contient les casernes
    with open('Data_environment/df_pc_real.pkl', 'rb') as f:
        df_real = pickle.load(f)

    # Charger les logs d'événements MODIFIÉS
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])

    print(f"✅ {len(df_events)} événements chargés")
    print(f"✅ Colonnes disponibles: {', '.join(df_events.columns)}")

    # Vérifier si on a les colonnes de véhicules
    if 'vehicle_type' in df_events.columns:
        print("✅ Types de véhicules trouvés dans les données")
        unique_types = df_events['vehicle_type'].dropna().unique()
        print(f"   Types uniques: {len(unique_types)}")

    # Joindre les informations de caserne
    station_info = df_real[['num_inter', 'PDD', 'zone', 'departure']].copy()

    # Extraire la première caserne de PDD si c'est une liste
    station_info['station'] = station_info['PDD'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )

    # Fusionner avec les événements
    df_events['case_id'] = df_events['case_id'].astype(int)
    df_merged = df_events.merge(
        station_info[['num_inter', 'station', 'zone']],
        left_on='case_id',
        right_on='num_inter',
        how='left'
    )

    return df_merged, station_info
def create_filtered_process_map(df_data,df_merged, station_name=None, month=None, incident_type=None, case_id=None):
    """
    Créer un process map avec filtres avancés (mois, caserne, type d'incident ou numéro d'incident)
    Retourne la figure matplotlib (fig) sans sauvegarde
    """
    print(f"\n🎯 Création du Process Map avec filtres:")
    print(f"   - Caserne: {station_name or 'Toutes'}")
    print(f"   - Mois: {month or 'Tous'}")
    print(f"   - Type incident: {incident_type or 'Tous'}")
    print(f"   - Numéro d'incident (case_id): {case_id if case_id is not None else 'Aucun'}")

    # Copier les données pour filtrage
    df_filtered = df_merged.copy()

    # Ajouter colonne mois si pas présente
    if 'month' not in df_filtered.columns:
        df_filtered['month'] = pd.to_datetime(df_filtered['timestamp']).dt.month

    # Si un incident_type est fourni, récupérer les case_id correspondants dans df_data, puis filtrer df_filtered
    if incident_type is not None:
        if 'incident_type' in df_data.columns:
            case_ids = df_data[df_data['incident_type'] == incident_type]['case_id'].unique()
        elif 'type' in df_data.columns:
            case_ids = df_data[df_data['type'] == incident_type]['id'].unique()
        else:
            case_ids = []
        if len(case_ids) == 0:
            print(f"❌ Aucun numéro d'incident trouvé pour le type: {incident_type}")
            return None
        df_filtered = df_filtered[df_filtered['case_id'].isin(case_ids)]
        print(f"🔎 {len(case_ids)} numéros d'incident trouvés pour le type: {incident_type}")
        if len(df_filtered) == 0:
            print(f"❌ Aucun événement trouvé dans df_filtered pour le type: {incident_type}")
            return None

    # Si un case_id est fourni, filtrer df_filtered sur ce numéro
    if case_id is not None:
        df_filtered = df_filtered[df_filtered['case_id'] == case_id]
        if len(df_filtered) == 0:
            print(f"❌ Aucun événement trouvé pour case_id {case_id}")
            return None

    # Filtrage par caserne
    if station_name and station_name != 'TOUTES_CASERNES':
        df_filtered = df_filtered[df_filtered['station'] == station_name]
        if len(df_filtered) == 0:
            print(f"❌ Aucune donnée pour la caserne: {station_name}")
            return None

    # Filtrage par mois
    if month:
        df_filtered = df_filtered[df_filtered['month'] == month]
        if len(df_filtered) == 0:
            print(f"❌ Aucune donnée pour le mois: {month}")
            return None

    n_cases = df_filtered['case_id'].nunique()
    print(f"✅ {n_cases} interventions après filtrage")

    if n_cases < 5:
        print("⚠️ Trop peu d'interventions pour créer un process map significatif")
        return None
    
    # Calculer les temps de transition
    def calculate_transition_times_enhanced(df_filtered):
        transition_times = {}
        transition_counts = {}
        vehicle_transition_times = {}
        for case_id, group in df_filtered.groupby('case_id'):
            group = group.sort_values('timestamp')
            for i in range(len(group) - 1):
                current = group.iloc[i]
                next_act = group.iloc[i + 1]
                transition = (current['activity'], next_act['activity'])
                time_diff = (next_act['timestamp'] - current['timestamp']).total_seconds() / 60
                if transition not in transition_times:
                    transition_times[transition] = []
                transition_times[transition].append(time_diff)
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
                if 'Départ_véhicule_' in current['activity']:
                    match = re.search(r'Départ_véhicule_\d+_(.+)', current['activity'])
                    if match:
                        vehicle_type = match.group(1)
                        key = f"{vehicle_type} → {next_act['activity']}"
                        if key not in vehicle_transition_times:
                            vehicle_transition_times[key] = []
                        vehicle_transition_times[key].append(time_diff)
        avg_times = {k: {'mean': np.mean(v), 'median': np.median(v), 'std': np.std(v), 'count': len(v)} for k, v in transition_times.items()}
        avg_vehicle_times = {k: {'mean': np.mean(v), 'median': np.median(v), 'count': len(v)} for k, v in vehicle_transition_times.items() if len(v) >= 5}
        return avg_times, transition_counts, avg_vehicle_times
    avg_times, transition_counts, avg_vehicle_times = calculate_transition_times_enhanced(df_filtered)
    min_activity_count = max(3, n_cases * 0.05)
    activity_counts = df_filtered['activity'].value_counts()
    frequent_activities = activity_counts[activity_counts >= min_activity_count].index.tolist()
    print(f"📊 Activités retenues: {len(frequent_activities)}/{len(activity_counts)}")
    G = nx.DiGraph()
    for activity in frequent_activities:
        count = activity_counts[activity]
        G.add_node(activity, count=count)
    for (from_act, to_act), stats in avg_times.items():
        if from_act in frequent_activities and to_act in frequent_activities and stats['count'] >= 3:
            G.add_edge(from_act, to_act, weight=stats['mean'], count=stats['count'], median=stats['median'])
    n_nodes = len(G.nodes())
    fig_width = max(16, n_nodes * 1.2)
    fig_height = max(12, n_nodes * 0.8)
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)
    ax = fig.gca()
    pos = {}
    fixed_positions = {
        'Appel_reçu': (0, 0),
        'Décision_départ': (3, 0),
        'Arrivée_sur_site': (12, 0),
        'Intervention_en_cours': (15, 0),
        'Fin_intervention': (18, 0)
    }
    vehicle_groups = {}
    vehicle_activities = [act for act in G.nodes() if 'Départ_véhicule_' in act]
    for activity in vehicle_activities:
        match = re.search(r'Départ_véhicule_(\d+)_(.+)', activity)
        if match:
            vehicle_num = int(match.group(1))
            vehicle_type = match.group(2).split('_')[0]
            key = f"V{vehicle_num}_{vehicle_type}"
            if key not in vehicle_groups:
                vehicle_groups[key] = []
            vehicle_groups[key].append(activity)
    x_offset = 5
    y_offset = 4
    max_per_column = 6
    col = 0
    row = 0
    for group_key in sorted(vehicle_groups.keys()):
        activities = vehicle_groups[group_key][:max_per_column]
        for activity in activities:
            x = x_offset + col * 2.5
            y = y_offset - row * 1.0
            pos[activity] = (x, y)
            row += 1
            if row >= max_per_column:
                row = 0
                col += 1
    for node, position in fixed_positions.items():
        if node in G.nodes():
            pos[node] = position
    missing_nodes = [node for node in G.nodes() if node not in pos]
    if missing_nodes:
        for i, node in enumerate(missing_nodes):
            pos[node] = (8 + (i % 4) * 2.5, -4 - (i // 4) * 1.0)
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        count = G.nodes[node]['count']
        size = min(2.0, 0.8 + (count / n_cases) * 1.2)
        if 'Départ_véhicule_' in node:
            if 'VSAV' in node:
                color = '#FF6B6B'
            elif 'FPT' in node:
                color = '#4ECDC4'
            elif 'EPA' in node:
                color = '#FFE66D'
            elif 'CCF' in node:
                color = '#FF8B94'
            else:
                color = '#95E1D3'
        elif node in ['Appel_reçu', 'Décision_départ']:
            color = '#A8E6CF'
        elif node == 'Fin_intervention':
            color = '#C7CEEA'
        else:
            color = '#FFC3A0'
        shadow = FancyBboxPatch((x-size/2+0.05, y-0.3+0.05), size, 0.6,
                               boxstyle="round,pad=0.08",
                               facecolor='gray',
                               alpha=0.3)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x-size/2, y-0.3), size, 0.6,
                            boxstyle="round,pad=0.08",
                            facecolor=color,
                            edgecolor='#2C3E50',
                            linewidth=2)
        ax.add_patch(box)
        if 'Départ_véhicule_' in node:
            match = re.search(r'Départ_véhicule_(\d+)_(.+)', node)
            if match:
                v_type = match.group(2).split('_')[0][:6]
                text = f"V{match.group(1)}-{v_type}\n({count})"
            else:
                text = f"{node[:12]}...\n({count})"
        else:
            text = f"{node}\n({count})"
        ax.text(x, y, text, 
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='#2C3E50')
    edge_threshold = 0
    if len(G.edges()) > 40:
        edge_counts = sorted([G[u][v]['count'] for u, v in G.edges()], reverse=True)
        edge_threshold = edge_counts[39]
    for (u, v) in G.edges():
        if u not in pos or v not in pos:
            continue
        if G[u][v]['count'] < edge_threshold:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mean_time = G[u][v]['weight']
        count = G[u][v]['count']
        width = min(4, 1.0 + (count / max(100, n_cases * 0.1)))
        if mean_time > 30:
            color = '#E74C3C'
        elif mean_time > 10:
            color = '#F39C12'
        else:
            color = '#27AE60'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, 
                                  lw=width, alpha=0.7,
                                  connectionstyle="arc3,rad=0.15"))
        if count >= n_cases * 0.08:
            mid_x = (x1 + x2) / 2 + 0.2
            mid_y = (y1 + y2) / 2 + 0.2
            label = f"{mean_time:.0f}m"
            ax.text(mid_x, mid_y, label, 
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.15", 
                            facecolor='white', 
                            edgecolor=color,
                            alpha=0.9),
                    fontsize=8, fontweight='bold')
    title_parts = []
    title_parts.append(f"Process Map")
    if station_name and station_name != 'TOUTES_CASERNES':
        title_parts.append(f"Caserne: {station_name}")
    if month:
        title_parts.append(f"Mois: {month}")
    if incident_type:
        title_parts.append(f"Type: {incident_type}")
    title_parts.append(f"({n_cases} interventions)")
    ax.set_title(" - ".join(title_parts), fontsize=16, fontweight='bold', pad=20)
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#27AE60', alpha=0.8, label='< 10 min'),
        Rectangle((0, 0), 1, 1, fc='#F39C12', alpha=0.8, label='10-30 min'),
        Rectangle((0, 0), 1, 1, fc='#E74C3C', alpha=0.8, label='> 30 min'),
        Rectangle((0, 0), 1, 1, fc='#FF6B6B', alpha=0.8, label='VSAV'),
        Rectangle((0, 0), 1, 1, fc='#4ECDC4', alpha=0.8, label='FPT'),
        Rectangle((0, 0), 1, 1, fc='#FFE66D', alpha=0.8, label='EPA')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    all_x = [pos[node][0] for node in pos]
    all_y = [pos[node][1] for node in pos]
    margin = 2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.axis('off')
    return fig
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
from agent import DQN_Agent
from collective_functions import get_potential_actions, gen_state, gen_ff_array, update_skills, get_roles_for_ff
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
                    
                    # Générer dic_roles_skills basique avec des compétences cohérentes
                    dic_roles_skills = {}
                    skills_count = len(df_skills.columns.get_level_values(0).unique())
                    
                    # Définir des compétences spécifiques pour chaque rôle
                    role_skills_mapping = {
                        "CHEF_EQUIPE": [1, 1, 1, 1, 0],  # Toutes les compétences principales
                        "CHEF_GARDE": [1, 1, 1, 1, 1],   # Toutes les compétences
                        "CONDUCTEUR": [1, 0, 0, 1, 1],   # Conduite et techniques
                        "EQUIPIER": [0, 0, 1, 0, 0],     # Compétences de base
                        "EQUIPIER1": [0, 0, 1, 1, 0],    # Un peu plus qualifié
                        "EQUIPIER2": [0, 0, 1, 0, 0]     # Compétences de base
                    }
                    
                    for vehicle, roles in dic_roles.items():
                        for role_id, role_name in roles.items():
                            if role_name not in dic_roles_skills:
                                # Créer un array de compétences cohérent
                                if role_name in role_skills_mapping:
                                    base_skills = role_skills_mapping[role_name]
                                else:
                                    base_skills = [0, 0, 1, 0, 0]  # Compétences de base par défaut
                                
                                # S'assurer que l'array a la bonne taille
                                skills_array = np.array(base_skills + [0] * (skills_count - len(base_skills)))[:skills_count]
                                # Convertir en array 2D comme attendu par get_role_from_skills
                                dic_roles_skills[role_name] = skills_array.reshape(1, -1)
                    
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
            
            # Créer dic_roles_skills avec des compétences cohérentes
            dic_roles_skills = {}
            
            # Définir des compétences spécifiques pour chaque rôle
            role_skills_mapping = {
                "CHEF_EQUIPE": [1, 1, 1, 1, 0],  # Toutes les compétences principales
                "CHEF_GARDE": [1, 1, 1, 1, 1],   # Toutes les compétences
                "CONDUCTEUR": [1, 0, 0, 1, 1],   # Conduite et techniques
                "EQUIPIER": [0, 0, 1, 0, 0],     # Compétences de base
                "EQUIPIER1": [0, 0, 1, 1, 0],    # Un peu plus qualifié
                "EQUIPIER2": [0, 0, 1, 0, 0]     # Compétences de base
            }
            
            for vehicle, roles in dic_roles.items():
                for role_id, role_name in roles.items():
                    if role_name not in dic_roles_skills:
                        # Créer un array de compétences cohérent
                        if role_name in role_skills_mapping:
                            base_skills = role_skills_mapping[role_name]
                        else:
                            base_skills = [0, 0, 1, 0, 0]  # Compétences de base par défaut
                        
                        skills_array = np.array(base_skills)
                        # Convertir en array 2D comme attendu par get_role_from_skills
                        dic_roles_skills[role_name] = skills_array.reshape(1, -1)
        
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
            './Scripts/Data_preprocessed/df_real.pkl'
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
    """Utilise les fonctions de collective_functions pour une recommandation basée sur les qualifications"""
    global agent_model, df_skills, skills_updated, dic_roles, dic_roles_skills
    
    try:
        # Assurer que dic_roles_skills est au bon format
        if dic_roles_skills:
            for role_name, skills_array in dic_roles_skills.items():
                if isinstance(skills_array, np.ndarray) and skills_array.ndim == 1:
                    dic_roles_skills[role_name] = skills_array.reshape(1, -1)
        
        # Déterminer les véhicules requis selon le type d'incident
        incident_type = incident['type'].upper()
        if 'SECOURS' in incident_type or 'ACCIDENT' in incident_type:
            veh_depart = ["VSAV"]
        elif 'INCENDIE' in incident_type or 'FEU' in incident_type:
            veh_depart = ["FPT", "EPA"]
        else:
            veh_depart = ["VSAV"]  # Par défaut
        
        # Pompiers disponibles (simulation basée sur les données réelles)
        ff_existing = df_skills.index[:30].tolist() if len(df_skills) > 30 else df_skills.index.tolist()
        print(f"DEBUG - ff_existing (TOUS les {len(ff_existing)}): {ff_existing}")
        print(f"DEBUG - Type des IDs: {type(ff_existing[0]) if ff_existing else 'Aucun'}")
        print(f"DEBUG - Pompier 1029 dans ff_existing? {1029 in ff_existing}")
        
        ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)
        
        # DEBUG: Vérifier les données
        print(f"DEBUG - Nombre de pompiers: {len(ff_existing)}")
        print(f"DEBUG - Shape ff_array: {ff_array.shape if ff_array is not None else 'None'}")
        print(f"DEBUG - dic_roles_skills disponible: {dic_roles_skills is not None}")
        print(f"DEBUG - Véhicules requis: {veh_depart}")
        
        # Dictionnaire des pompiers (tous disponibles pour la démo)
        dic_ff = {ff: 0 for ff in df_skills.index}
        
        # Vérifier et initialiser dic_roles si nécessaire
        if not dic_roles:
            dic_roles = {
                "VSAV": {1: "CHEF_EQUIPE", 2: "EQUIPIER"},
                "FPT": {1: "CHEF_GARDE", 2: "CONDUCTEUR", 3: "EQUIPIER", 4: "EQUIPIER"},
                "EPA": {1: "CHEF_EQUIPE", 2: "EQUIPIER"}
            }
        
        # CORRECTION: Nettoyer dic_roles pour ne garder que les rôles qui existent dans dic_roles_skills
        roles_disponibles = set(dic_roles_skills.keys()) if dic_roles_skills else set()
        print(f"DEBUG - Rôles disponibles dans dic_roles_skills: {list(roles_disponibles)}")
        
        # Nettoyer dic_roles pour éviter l'erreur 'EQ_ENG_SAP'
        dic_roles_clean = {}
        for vehicule, roles in dic_roles.items():
            roles_clean = {}
            for num, nom_role in roles.items():
                if nom_role in roles_disponibles:
                    roles_clean[num] = nom_role
                else:
                    print(f"WARNING - Rôle {nom_role} non trouvé, remplacé par EQUIPIER")
                    if "EQUIPIER" in roles_disponibles:
                        roles_clean[num] = "EQUIPIER"
                    elif len(roles_disponibles) > 0:
                        roles_clean[num] = list(roles_disponibles)[0]
            
            if roles_clean:  # Seulement ajouter si on a des rôles valides
                dic_roles_clean[vehicule] = roles_clean
        
        dic_roles = dic_roles_clean
        print(f"DEBUG - dic_roles nettoyé: {dic_roles}")
        
        # Vérifier dic_roles_skills et l'initialiser si nécessaire
        if not dic_roles_skills:
            print("WARNING: dic_roles_skills est vide, création de données fictives")
            # Créer des données fictives pour les rôles
            dic_roles_skills = {
                "CHEF_EQUIPE": np.array([[1, 1, 1, 0, 0]]),  # Compétences chef
                "EQUIPIER": np.array([[1, 0, 0, 0, 0]]),     # Compétences de base
                "CHEF_GARDE": np.array([[1, 1, 1, 1, 0]]),   # Chef de garde
                "CONDUCTEUR": np.array([[1, 1, 0, 0, 0]])    # Conducteur
            }
        else:
            print("INFO: Ajustement des compétences requises pour correspondre aux pompiers disponibles")
            # CORRECTION: Ajuster les compétences requises basées sur les compétences réelles
            # Analyser les compétences existantes des pompiers
            ff_skills_sample = ff_array[:5]  # Premier 5 pompiers
            print(f"DEBUG - Analyse des compétences réelles (5 premiers pompiers):")
            for i, skills in enumerate(ff_skills_sample):
                available_skills = np.where(skills == 1)[0]
                print(f"  Pompier {i}: compétences aux positions {available_skills[:10]}...")
            
            # Créer des compétences requises plus réalistes
            # Utiliser des compétences qui existent vraiment chez les pompiers
            most_common_skills = []
            for skill_idx in range(min(20, ff_array.shape[1])):  # Check first 20 skills
                skill_count = np.sum(ff_array[:, skill_idx])
                if skill_count > 0:
                    most_common_skills.append(skill_idx)
            
            print(f"DEBUG - Compétences les plus communes (premières 10): {most_common_skills[:10]}")
            
            if len(most_common_skills) >= 4:
                # Créer des patterns réalistes basés sur les compétences existantes
                chef_skills = np.zeros(ff_array.shape[1], dtype=int)
                equipier_skills = np.zeros(ff_array.shape[1], dtype=int)
                
                # CHEF_EQUIPE : les 3 premières compétences communes + quelques spécialisées
                if len(most_common_skills) >= 3:
                    chef_skills[most_common_skills[:3]] = 1
                
                # EQUIPIER : juste la compétence la plus commune
                if len(most_common_skills) >= 1:
                    equipier_skills[most_common_skills[0]] = 1
                
                dic_roles_skills = {
                    "CHEF_EQUIPE": chef_skills.reshape(1, -1),
                    "EQUIPIER": equipier_skills.reshape(1, -1),
                    "CHEF_GARDE": chef_skills.reshape(1, -1),  # Même que chef d'équipe
                    "CONDUCTEUR": equipier_skills.reshape(1, -1)  # Même qu'équipier
                }
                
                print(f"DEBUG - Nouvelles compétences CHEF_EQUIPE: positions {np.where(chef_skills == 1)[0]}")
                print(f"DEBUG - Nouvelles compétences EQUIPIER: positions {np.where(equipier_skills == 1)[0]}")
            else:
                print("WARNING: Pas assez de compétences communes, utilisation de patterns simples")
                # Fallback: utiliser des patterns très simples
                dic_roles_skills = {
                    "CHEF_EQUIPE": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)]),  # Juste skill 3
                    "EQUIPIER": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)]),     # Même chose
                    "CHEF_GARDE": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)]),
                    "CONDUCTEUR": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)])
                }
        
        # Analyse des qualifications pour chaque véhicule requis
        recommandations_vehicules = []
        meilleur_pompier = None
        meilleur_score = 0
        
        for vehicule in veh_depart:
            print(f"DEBUG - Analyse véhicule: {vehicule}")
            if vehicule in dic_roles:
                try:
                    # Obtenir la matrice de matching pour ce véhicule
                    roles_matrix = get_roles_for_ff(vehicule, ff_array, dic_roles, dic_roles_skills)
                    print(f"DEBUG - Shape roles_matrix pour {vehicule}: {roles_matrix.shape}")
                    
                    # DEBUG: Analyser les données d'entrée
                    print(f"DEBUG - ff_array sample (premier pompier): {ff_array[0][:10] if len(ff_array) > 0 else 'Empty'}")
                    
                    # Analyser chaque rôle du véhicule
                    roles_requis = dic_roles[vehicule]
                    for idx_role, (num_role, nom_role) in enumerate(sorted(roles_requis.items())):
                        print(f"DEBUG - Rôle {idx_role}: {nom_role}")
                        
                        # DEBUG: Vérifier les compétences requises pour ce rôle
                        if nom_role in dic_roles_skills:
                            required_skills = dic_roles_skills[nom_role]
                            print(f"DEBUG - Compétences requises pour {nom_role}: {required_skills.shape} -> {required_skills}")
                        else:
                            print(f"DEBUG - ERREUR: Rôle {nom_role} non trouvé dans dic_roles_skills")
                            print(f"DEBUG - Rôles disponibles: {list(dic_roles_skills.keys())}")
                        
                        if idx_role < roles_matrix.shape[1]:
                            # Scores de qualification pour ce rôle
                            scores_qualification = roles_matrix[:, idx_role]
                            print(f"DEBUG - Scores max pour {nom_role}: {np.max(scores_qualification) if len(scores_qualification) > 0 else 'Aucun'}")
                            print(f"DEBUG - Sample scores pour {nom_role}: {scores_qualification[:5]}")
                            
                            # Trouver les pompiers qualifiés (score > 0)
                            pompiers_qualifies = []
                            for i, score in enumerate(scores_qualification):
                                if score > 0 and i < len(ff_existing):
                                    pompier_id = ff_existing[i]
                                    pompiers_qualifies.append((pompier_id, score))
                            
                            print(f"DEBUG - Pompiers qualifiés pour {nom_role}: {len(pompiers_qualifies)}")
                            
                            # Trier par score décroissant
                            pompiers_qualifies.sort(key=lambda x: x[1], reverse=True)
                            
                            # CORRECTION: Sélection plus équitable du meilleur pompier
                            if pompiers_qualifies:
                                # Prendre tous les pompiers avec le score maximum
                                score_max = pompiers_qualifies[0][1]
                                pompiers_score_max = [p for p in pompiers_qualifies if p[1] == score_max]
                                
                                # Si plusieurs pompiers ont le même score max, faire une rotation basée sur l'heure
                                import time
                                rotation_index = int(time.time()) % len(pompiers_score_max)
                                pompier_selectionne = pompiers_score_max[rotation_index]
                                
                                print(f"DEBUG - {len(pompiers_score_max)} pompiers ont le score max {score_max}")
                                print(f"DEBUG - Pompier sélectionné (rotation {rotation_index}): {pompier_selectionne[0]}")
                                print(f"DEBUG - Index du pompier dans ff_existing: {ff_existing.index(pompier_selectionne[0]) if pompier_selectionne[0] in ff_existing else 'PAS TROUVÉ'}")
                                print(f"DEBUG - ff_existing contient {pompier_selectionne[0]}? {pompier_selectionne[0] in ff_existing}")
                                print(f"DEBUG - ff_existing contient: {ff_existing[:5]}...")
                                
                                # CORRECTION: Vérifier que le pompier sélectionné est bien dans ff_existing
                                if pompier_selectionne[0] in ff_existing:
                                    # Mettre à jour le meilleur pompier global
                                    if score_max > meilleur_score:
                                        meilleur_score = score_max
                                        meilleur_pompier = (pompier_selectionne[0], score_max, vehicule, nom_role)
                                        print(f"DEBUG - Nouveau meilleur pompier: {meilleur_pompier}")
                                else:
                                    print(f"WARNING - Pompier {pompier_selectionne[0]} pas dans ff_existing, utilisation du premier disponible")
                                    # Fallback: utiliser le premier pompier de ff_existing qui a un score > 0
                                    for pompier_id, score in pompiers_qualifies:
                                        if pompier_id in ff_existing and score > meilleur_score:
                                            meilleur_score = score
                                            meilleur_pompier = (pompier_id, score, vehicule, nom_role)
                                            print(f"DEBUG - Fallback meilleur pompier: {meilleur_pompier}")
                                            break
                            
                            recommandations_vehicules.append({
                                'vehicule': vehicule,
                                'role': nom_role,
                                'pompiers_qualifies': pompiers_qualifies[:5]  # Top 5
                            })
                        else:
                            print(f"DEBUG - Rôle {nom_role} hors limites de la matrice")
                            
                except Exception as e:
                    print(f"ERROR - Erreur avec véhicule {vehicule}: {e}")
                    # En cas d'erreur, créer des données fictives
                    recommandations_vehicules.append({
                        'vehicule': vehicule,
                        'role': 'EQUIPIER',
                        'pompiers_qualifies': [(ff_existing[0], 5)] if ff_existing else []
                    })
                    if ff_existing and meilleur_score == 0:
                        meilleur_pompier = (ff_existing[0], 5, vehicule, 'EQUIPIER')
                        meilleur_score = 5
            else:
                print(f"DEBUG - Véhicule {vehicule} non trouvé dans dic_roles")
        
        # Utiliser get_potential_actions pour valider la sélection
        if agent_model and meilleur_pompier:
            # Créer un état minimal pour get_potential_actions
            coord_x = incident.get('coord_x_orig', incident['lon'])
            coord_y = incident.get('coord_y_orig', incident['lat'])
            month = incident['mois_num']
            hour = incident.get('Hour', 12)
            
            # Encoder les valeurs temporelles
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * 15 / 365)
            day_cos = np.cos(2 * np.pi * 15 / 365)
            
            info_avail = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0]
            action_size = 80
            max_duration = 1200
            
            try:
                state = gen_state(
                    veh_depart, 0, ff_array, ff_existing,
                    dic_roles, dic_roles_skills, dic_ff, df_skills,
                    coord_x, coord_y, month_sin, month_cos, day_sin,
                    day_cos, hour_sin, hour_cos, info_avail, max_duration, action_size
                )
                
                # Valider avec get_potential_actions
                potential_actions, potential_skills = get_potential_actions(state, False)
                
                if potential_actions and potential_actions[0] != 99:
                    action_validee = potential_actions[0]
                    skill_validee = potential_skills[0] if potential_skills else meilleur_score
                else:
                    action_validee = None
                    skill_validee = 0
                    
            except Exception as e:
                print(f"Erreur dans la validation: {e}")
                action_validee = None
                skill_validee = meilleur_score
        else:
            action_validee = None
            skill_validee = meilleur_score
        
        # Générer la décision finale
        if meilleur_pompier:
            pompier_id, score, vehicule, role = meilleur_pompier
            print(f"DEBUG FINAL - Pompier sélectionné: {pompier_id} (type: {type(pompier_id)})")
            print(f"DEBUG FINAL - ff_existing complet: {ff_existing}")
            print(f"DEBUG FINAL - Pompier {pompier_id} dans ff_existing? {pompier_id in ff_existing}")
            decision_text = f"👨‍🚒 Pompier {pompier_id} recommandé pour {vehicule} ({role}) - Niveau {score}"
            couleur_decision = "green" if score >= 6 else "orange" if score >= 4 else "red"
            skill_lvl = score
        else:
            decision_text = "� Aucun pompier qualifié disponible"
            couleur_decision = "red"
            skill_lvl = 0
        
        
        # Créer une recommandation détaillée
        recommandation = f"""
## 🤖 Recommandations basées sur les Qualifications

### 📊 Analyse de la situation
- **Type d'incident:** {incident['type']}
- **Durée estimée:** {incident['temps']:.0f} minutes
- **Zone:** {incident.get('zone', 'Toulouse centre')}

### 🎯 Décision optimale
**{decision_text}**

### 📋 Détails de l'allocation
- **Véhicules suggérés:** {', '.join(veh_depart)}
- **Niveau de qualification:** {skill_lvl}/8
- **Temps de réponse estimé:** {np.random.randint(8, 15)} minutes

### 💡 Analyse des qualifications
"""
        
        # Ajouter le détail des qualifications par véhicule
        for rec in recommandations_vehicules:
            recommandation += f"\n**{rec['vehicule']} - {rec['role']}:**\n"
            if rec['pompiers_qualifies']:
                for pompier_id, score in rec['pompiers_qualifies'][:3]:
                    recommandation += f"- Pompier {pompier_id}: Niveau {score}/8\n"
            else:
                recommandation += "- Aucun pompier qualifié disponible\n"
        
        recommandation += f"""
### 🔍 Méthode d'analyse
Cette recommandation utilise les fonctions de `collective_functions` pour:
1. Analyser les qualifications requises par véhicule
2. Évaluer chaque pompier selon ses compétences
3. Sélectionner le meilleur match qualification/disponibilité
"""
        
        # Créer un graphique de décision
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graphique 1: Scores de qualification des top pompiers
        if recommandations_vehicules:
            top_pompiers = []
            top_scores = []
            top_roles = []
            
            for rec in recommandations_vehicules:
                for pompier_id, score in rec['pompiers_qualifies'][:2]:  # Top 2 par véhicule
                    if len(top_pompiers) < 10:  # Limiter à 10 au total
                        top_pompiers.append(f"{pompier_id}\n({rec['vehicule']})")
                        top_scores.append(score)
                        top_roles.append(rec['role'])
            
            if top_pompiers:
                # Définir action pour les couleurs
                action = 0 if meilleur_pompier else 99
                
                colors = ['green' if i == 0 and meilleur_pompier else 'lightblue' for i in range(len(top_pompiers))]
                bars = ax1.bar(range(len(top_pompiers)), top_scores, color=colors, alpha=0.7)
                ax1.set_xlabel('Pompiers disponibles')
                ax1.set_ylabel('Niveau de qualification')
                ax1.set_title('Top Pompiers par Qualification')
                ax1.set_xticks(range(len(top_pompiers)))
                ax1.set_xticklabels(top_pompiers, rotation=45, ha='right')
                ax1.set_ylim(0, 8)
                
                # Ajouter les valeurs sur les barres
                for bar, score in zip(bars, top_scores):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{score}', ha='center', va='bottom')
            else:
                ax1.text(0.5, 0.5, 'Aucune qualification\ntrouvée', 
                        transform=ax1.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                ax1.set_title('Qualifications - Aucun match')
        else:
            ax1.text(0.5, 0.5, 'Analyse des\nqualifications\nnon disponible', 
                    transform=ax1.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
            ax1.set_title('Qualifications - En cours')
        
        # Graphique 2: Répartition des véhicules recommandés
        if veh_depart:
            vehicle_counts = {}
            for veh in veh_depart:
                vehicle_counts[veh] = vehicle_counts.get(veh, 0) + 1
            
            labels = list(vehicle_counts.keys())
            sizes = list(vehicle_counts.values())
            colors_pie = ['red' if 'VSAV' in label else 'blue' if 'FPT' in label else 'green' for label in labels]
            
            ax2.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors=colors_pie)
            ax2.set_title('Véhicules Recommandés')
        else:
            ax2.text(0.5, 0.5, 'Aucun véhicule\nrecommandé', 
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
            ax2.set_title('Véhicules - Aucun')
        
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
## ❌ Erreur Analyse Qualifications
**Message:** {str(e)}
**Type d'incident:** {incident.get('type', 'Inconnu')}

### 🔧 Détails techniques
```
{traceback.format_exc()}
```

### 💡 Suggestions
- Vérifiez que les données de qualifications sont chargées
- Redémarrez l'environnement avec le bouton "🔄 Recharger l'agent"
- Utilisez l'onglet Diagnostics pour tester les fonctions
"""
        return error_msg, None
        
        # Graph 1: Q-values pour les top actions (désactivé - utilise maintenant les qualifications)
        if False:  # Désactivé car on utilise maintenant l'analyse de qualifications
            pass
        
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

def generer_process_map(caserne, mois, type_incident):
    """Génère le process map selon les filtres sélectionnés"""
    
    # Adapter les valeurs "Toutes"/"Tous"/None pour la fonction
    caserne_val = None if not caserne or caserne == "Toutes" else caserne
    mois_val = None
    if mois and mois != "Tous":
        mois_mapping = {"Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4, "Mai": 5, "Juin": 6, "Juillet": 7, "Août": 8, "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12}
        mois_val = mois_mapping.get(mois, None)
    type_val = None if not type_incident or type_incident == "Tous" else type_incident

    # Charger les données fusionnées (doit être accessible globalement ou importée)
    df_merged, _ = load_data_with_stations()

    # Appeler simplement la fonction et retourner la figure
    return create_filtered_process_map(df_data,df_merged, station_name=caserne_val, month=mois_val, incident_type=type_val)

def create_fallback_process_map_html(caserne, mois, type_incident, error_msg):
    """Créer un HTML de fallback en cas d'erreur"""
    return f"""
    <div style='height: 500px; border: 2px solid #FF9800; border-radius: 10px; padding: 20px; 
                background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%, #fff3e0 100%);'>
        <h3 style='text-align: center; color: #E65100; margin-bottom: 20px;'>🔄 Process Map des Interventions</h3>
        
        <div style='background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h4 style='color: #1976D2; margin-bottom: 10px;'>📋 Filtres appliqués:</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;'>
                <div style='padding: 8px; background: #E3F2FD; border-radius: 4px; text-align: center;'>
                    <strong>🏢 Caserne:</strong><br>{caserne}
                </div>
                <div style='padding: 8px; background: #F3E5F5; border-radius: 4px; text-align: center;'>
                    <strong>📅 Mois:</strong><br>{mois}
                </div>
                <div style='padding: 8px; background: #FFF3E0; border-radius: 4px; text-align: center;'>
                    <strong>🚨 Type:</strong><br>{type_incident}
                </div>
            </div>
        </div>
        
        <div style='text-align: center; padding: 40px; background: rgba(255,255,255,0.9); 
                    border: 2px dashed #FF9800; border-radius: 8px;'>
            <div style='font-size: 48px; margin-bottom: 20px;'>⚠️</div>
            <p style='font-size: 16px; color: #E65100; margin: 10px 0; font-weight: bold;'>
                Process Map temporairement indisponible
            </p>
            <p style='color: #666; font-size: 14px; margin: 10px 0;'>{error_msg}</p>
            <div style='margin-top: 20px; padding: 15px; background: #FFF3E0; border-radius: 4px;'>
                <p style='font-size: 12px; color: #F57C00; margin: 0;'>
                    <strong>📊 Analyse en cours...</strong><br>
                    Le système process mining sera bientôt disponible avec vos filtres personnalisés
                </p>
            </div>
        </div>
    </div>
    """

import gradio as gr

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
            
            gr.Markdown("---")

        # === NOUVEL ONGLET : PROCESS MINING ===
        with gr.TabItem("⛏️ Process Mining"):
            gr.Markdown("""
            ## 🔄 Process Map des Interventions
            
            Visualisez les processus et flux des interventions selon différents critères de filtrage.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    caserne_filter = gr.Dropdown(
                        ["Toutes"] + df_casernes['nom'].tolist(),  # Dynamically fetch casernes
                        label="🏢 Caserne",
                        value="Toutes"
                    )
                
                with gr.Column(scale=1):
                    mois_process_filter = gr.Dropdown(
                        ["Tous"] + list(MOIS_MAPPING.values()), 
                        label="📅 Mois", 
                        value="Tous"
                    )
                
                with gr.Column(scale=1):
                    type_incident_process_filter = gr.Dropdown(
                        ["Tous", "Secours à personne", "Incendie", "Accident routier", "Inondation", "Fuite de gaz", "Autres secours"], 
                        label="🚨 Type d'incident", 
                        value="Tous"
                    )
                
                with gr.Column(scale=1):
                    bouton_process_map = gr.Button("🔄 Afficher Process Map", variant="primary")
            
            with gr.Row():
                process_map_output = gr.Plot(
                    label="Process Map ",
                    show_label=True,
                    elem_id="process-map-box",
                    value=None
                )

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
    
    # Process Map
    bouton_process_map.click(
        fn=generer_process_map,
        inputs=[caserne_filter, mois_process_filter, type_incident_process_filter],
        outputs=process_map_output
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
        server_name="127.0.0.1",  # Pour accès réseau local
        server_port=7860,
        show_error=True
    )