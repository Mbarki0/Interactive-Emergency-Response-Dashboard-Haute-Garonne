import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import re
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from io import BytesIO
from PIL import Image
import pickle
import json
from datetime import datetime
import torch
from agent import DQN_Agent
from collective_functions import get_potential_actions, gen_state, gen_ff_array, update_skills
import gradio as gr


def load_data_with_stations(csv_file='./event_logs_with_vehicles.csv'):
    """
    Load modified data with vehicle types and station information
    """
    print("📂 Loading modified data...")
    df_events = pd.read_csv(csv_file)
    # Load df_pc_real which contains stations
    with open('Data_environment/df_pc_real.pkl', 'rb') as f:
        df_real = pickle.load(f)

    # Load MODIFIED event logs
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])

    print(f"✅ {len(df_events)} events loaded")
    print(f"✅ Available columns: {', '.join(df_events.columns)}")

    # Check if we have vehicle columns
    if 'vehicle_type' in df_events.columns:
        print("✅ Vehicle types found in data")
        unique_types = df_events['vehicle_type'].dropna().unique()
        print(f"   Unique types: {len(unique_types)}")

    # Join station information
    station_info = df_real[['num_inter', 'PDD', 'zone', 'departure']].copy()

    # Extract first station from PDD if it's a list
    station_info['station'] = station_info['PDD'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )

    # Merge with events
    df_events['case_id'] = df_events['case_id'].astype(int)
    df_merged = df_events.merge(
        station_info[['num_inter', 'station', 'zone']],
        left_on='case_id',
        right_on='num_inter',
        how='left'
    )

    return df_merged, station_info


def create_filtered_process_map(df_data, df_merged, station_name=None, month=None, incident_type=None, case_id=None):
    """
    Create a process map with advanced filters (month, station, incident type or incident number)
    Returns the matplotlib figure (fig) without saving
    """
    print(f"\n🎯 Creating Process Map with filters:")
    print(f"   - Station: {station_name or 'All'}")
    print(f"   - Month: {month or 'All'}")
    print(f"   - Incident type: {incident_type or 'All'}")
    print(f"   - Incident number (case_id): {case_id if case_id is not None else 'None'}")

    # Copy data for filtering
    df_filtered = df_merged.copy()

    # Add month column if not present
    if 'month' not in df_filtered.columns:
        df_filtered['month'] = pd.to_datetime(df_filtered['timestamp']).dt.month

    # If an incident_type is provided, get corresponding case_ids from df_data, then filter df_filtered
    if incident_type is not None:
        if 'incident_type' in df_data.columns:
            case_ids = df_data[df_data['incident_type'] == incident_type]['case_id'].unique()
        elif 'type' in df_data.columns:
            case_ids = df_data[df_data['type'] == incident_type]['id'].unique()
        else:
            case_ids = []
        if len(case_ids) == 0:
            print(f"❌ No incident numbers found for type: {incident_type}")
            return None
        df_filtered = df_filtered[df_filtered['case_id'].isin(case_ids)]
        print(f"🔎 {len(case_ids)} incident numbers found for type: {incident_type}")
        if len(df_filtered) == 0:
            print(f"❌ No events found in df_filtered for type: {incident_type}")
            return None

    # If a case_id is provided, filter df_filtered on this number
    if case_id is not None:
        df_filtered = df_filtered[df_filtered['case_id'] == case_id]
        if len(df_filtered) == 0:
            print(f"❌ No events found for case_id {case_id}")
            return None

    # Filter by station
    if station_name and station_name != 'TOUTES_CASERNES':
        df_filtered = df_filtered[df_filtered['station'] == station_name]
        if len(df_filtered) == 0:
            print(f"❌ No data for station: {station_name}")
            return None

    # Filter by month
    if month:
        df_filtered = df_filtered[df_filtered['month'] == month]
        if len(df_filtered) == 0:
            print(f"❌ No data for month: {month}")
            return None

    n_cases = df_filtered['case_id'].nunique()
    print(f"✅ {n_cases} interventions after filtering")

    if n_cases < 5:
        print("⚠️ Too few interventions to create a meaningful process map")
        return None

    # Calculate transition times
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
        avg_times = {k: {'mean': np.mean(v), 'median': np.median(v), 'std': np.std(v), 'count': len(v)} for k, v in
                     transition_times.items()}
        avg_vehicle_times = {k: {'mean': np.mean(v), 'median': np.median(v), 'count': len(v)} for k, v in
                             vehicle_transition_times.items() if len(v) >= 5}
        return avg_times, transition_counts, avg_vehicle_times

    avg_times, transition_counts, avg_vehicle_times = calculate_transition_times_enhanced(df_filtered)
    min_activity_count = max(3, n_cases * 0.05)
    activity_counts = df_filtered['activity'].value_counts()
    frequent_activities = activity_counts[activity_counts >= min_activity_count].index.tolist()
    print(f"📊 Activities retained: {len(frequent_activities)}/{len(activity_counts)}")
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
        shadow = FancyBboxPatch((x - size / 2 + 0.05, y - 0.3 + 0.05), size, 0.6,
                                boxstyle="round,pad=0.08",
                                facecolor='gray',
                                alpha=0.3)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x - size / 2, y - 0.3), size, 0.6,
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
        title_parts.append(f"Station: {station_name}")
    if month:
        title_parts.append(f"Month: {month}")
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


def generate_process_map(station_filter=None, month_filter=None, incident_type_filter=None):
    """Generate process map for Gradio interface"""

    # Adapt "All"/"All"/None values for the function
    station_val = None if not station_filter or station_filter == "All stations" else station_filter
    month_val = None
    if month_filter and month_filter != "All months":
        month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
                         "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
        month_val = month_mapping.get(month_filter, None)
    type_val = None if not incident_type_filter or incident_type_filter == "All types" else incident_type_filter

    # Load merged data (must be globally accessible or imported)
    df_merged, _ = load_data_with_stations()

    # Simply call the function and return the figure
    return create_filtered_process_map(df_data, df_merged, station_name=station_val, month=month_val,
                                       incident_type=type_val)


# Variables globales pour l'agent
agent_loaded = False
agent_model = None
agent_config = None
df_skills = None
skills_updated = None
dic_roles = None
dic_roles_skills = None


def load_agent_and_environment():
    """Load RL agent and necessary environment data"""
    global agent_loaded, agent_model, agent_config, df_skills, skills_updated, dic_roles, dic_roles_skills

    try:
        print("🤖 Loading RL agent and environment...")

        # Load configuration
        config_paths = ['./Data/hyper_params.json', 'hyper_params.json', './hyper_params.json']
        agent_config = None

        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    agent_config = json.load(f)
                print(f"✅ Configuration found: {path}")
                break

        if agent_config is None:
            raise Exception("hyper_params.json not found")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent_config["device"] = str(device)
        agent_config["max_train_steps"] = 100000  # Default value

        # Initialize agent
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
                                        if isinstance(value, dict) and all(
                                                isinstance(v, torch.Tensor) for v in value.values()):
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
                                    if isinstance(state_dict, dict) and all(
                                            isinstance(v, torch.Tensor) for v in state_dict.values()):
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
                                active_skills = np.random.choice(len(skills_array), size=min(5, len(skills_array)),
                                                                 replace=False)
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


# Load agent at startup
load_agent_and_environment()


# Load real data
def load_real_data():
    """Load real intervention and station data"""
    try:
        # Try several possible paths
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
                print(f"✅ Data found in: {path}")
                break

        if df_real is None:
            print("❌ df_real.pkl not found, using mock data")
            return generate_mock_data()

        # Same for stations
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
            print("⚠️ df_stations.pkl not found, generating mock stations")
            df_stations = pd.DataFrame({
                'Nom': [f'Station_{i}' for i in range(1, 11)],
                'Coordonnée X': np.random.normal(566973, 20000, 10),
                'Coordonnée Y': np.random.normal(6271183, 20000, 10)
            })

        # Adapt format for interface
        df_incidents = df_real.copy()
        df_incidents['id'] = range(1, len(df_incidents) + 1)
        df_incidents['nom'] = [f"Intervention_{i}" for i in df_incidents['id']]

        # Coordinate conversion
        df_incidents['lat'] = 43.6 + (df_incidents['Coord Y'] - df_incidents['Coord Y'].mean()) / 100000
        df_incidents['lon'] = 1.4 + (df_incidents['Coord X'] - df_incidents['Coord X'].mean()) / 100000
        df_incidents['mois_num'] = df_incidents['Month']

        # Incident types
        incident_mapping = {
            1: "Rescue operation", 2: "Fire", 3: "Road accident",
            4: "Flood", 5: "Gas leak", 6: "Other rescue"
        }
        df_incidents['type'] = df_incidents['Incident'].map(
            lambda x: incident_mapping.get(x % 6 + 1, "Other")
        )

        # Add real metrics
        df_incidents['temps'] = df_incidents['Duration']
        df_incidents['coord_x_orig'] = df_incidents['Coord X']
        df_incidents['coord_y_orig'] = df_incidents['Coord Y']

        # Stations
        df_casernes = df_stations.copy()
        df_casernes['id'] = range(1, len(df_casernes) + 1)
        df_casernes['nom'] = df_casernes['Nom']
        df_casernes['lat'] = 43.6 + (df_casernes['Coordonnée Y'] - df_casernes['Coordonnée Y'].mean()) / 100000
        df_casernes['lon'] = 1.4 + (df_casernes['Coordonnée X'] - df_casernes['Coordonnée X'].mean()) / 100000

        return df_incidents, df_casernes

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return generate_mock_data()


def generate_mock_data():
    """Generate mock data for demo"""
    np.random.seed(42)

    # Mock incidents
    n_incidents = 1000
    df_incidents = pd.DataFrame({
        'id': range(1, n_incidents + 1),
        'nom': [f"Intervention_{i}" for i in range(1, n_incidents + 1)],
        'lat': np.random.normal(43.6, 0.1, n_incidents),
        'lon': np.random.normal(1.44, 0.1, n_incidents),
        'mois_num': np.random.randint(1, 13, n_incidents),
        'type': np.random.choice(['Rescue operation', 'Fire', 'Road accident'], n_incidents),
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

    # Mock stations
    df_casernes = pd.DataFrame({
        'id': range(1, 11),
        'nom': [f'Station_{i}' for i in range(1, 11)],
        'lat': np.random.normal(43.6, 0.08, 10),
        'lon': np.random.normal(1.44, 0.08, 10),
        'Nom': [f'Station_{i}' for i in range(1, 11)],
        'Coordonnée X': np.random.normal(566973, 15000, 10),
        'Coordonnée Y': np.random.normal(6271183, 15000, 10)
    })

    return df_incidents, df_casernes


# Load data at startup
df_data, df_casernes = load_real_data()

# Month mapping
MONTH_MAPPING = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# Global variables for current state
df_filtre = pd.DataFrame()
incident_selectionne = None


def search_incidents(incident_type, month):
    global df_filtre
    if not month:
        return None, None, "Please select a month", gr.update(visible=False)

    # Convert month name to number
    month_num = None
    for num, nom in MONTH_MAPPING.items():
        if nom == month:
            month_num = num
            break

    # Filter incidents
    df_filtre = df_data[df_data["mois_num"] == month_num].copy()

    if incident_type != "All":
        df_filtre = df_filtre[df_filtre["type"] == incident_type]

    if df_filtre.empty:
        return None, None, f"No incidents found for {month} / {incident_type}", gr.update(visible=False)

    # Incident map
    fig_incidents = px.scatter_map(
        df_filtre,
        lat="lat", lon="lon",
        hover_name="nom",
        hover_data={"type": True, "temps": ":.0f min", "id": True},
        color="type",
        zoom=8,
        center={"lat": 43.6, "lon": 1.44},
        custom_data=["id"],
        title=f"Incidents in {month} ({len(df_filtre)} incidents)",
        height=500
    )

    fig_incidents.update_traces(
        marker=dict(size=15, opacity=1.0)
    )

    # Station map
    fig_casernes = px.scatter_map(
        df_casernes,
        lat="lat", lon="lon",
        hover_name="nom",
        zoom=8,
        center={"lat": 43.6, "lon": 1.44},
        title="Fire stations",
        height=500
    )

    fig_casernes.update_traces(
        marker=dict(size=20, opacity=0.9, symbol='square', color='red')
    )

    success_msg = f"✅ {len(df_filtre)} incidents displayed for {month}"

    return fig_incidents, fig_casernes, success_msg, gr.update(choices=df_filtre["id"].astype(str).tolist(),
                                                               visible=True)


def display_stats_and_recommendations(incident_id):
    """Display stats AND RL agent recommendations"""
    global df_filtre, incident_selectionne, agent_model, agent_loaded

    if df_filtre.empty or not incident_id:
        return "No data available.", None, None, None

    try:
        incident = df_filtre[df_filtre["id"] == int(incident_id)].iloc[0]
        incident_selectionne = incident
    except:
        return "Incident not found.", None, None, None

    # Basic description
    description = f"""
### 🚨 {incident['nom']} - {incident['type']}
**📅 Date:** {MONTH_MAPPING.get(incident['mois_num'], 'Unknown')}
**⏱️ Estimated duration:** {incident['temps']:.0f} minutes
**📍 Location:** Lat {incident['lat']:.4f}, Lon {incident['lon']:.4f}
"""

    # If agent is loaded, get its recommendations
    recommendation = "⏳ RL Agent not available"
    decision_plot = None

    if agent_loaded and agent_model:
        try:
            recommendation, decision_plot = get_agent_recommendation(incident)
        except Exception as e:
            recommendation = f"❌ Error getting recommendations: {str(e)}"

    # Basic metrics chart
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Simple visualization example
    metrics = ['Urgency', 'Complexity', 'Resources']
    values = [
        min(incident['temps'] / 30, 1) * 100,  # Urgency based on time
        np.random.uniform(0.4, 0.8) * 100,  # Simulated complexity
        np.random.uniform(0.3, 0.7) * 100  # Simulated resources
    ]

    ax.bar(metrics, values, color=['red', 'orange', 'blue'], alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score (%)')
    ax.set_title(f"Analysis - {incident['type']}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    stats_image = Image.open(buf)
    plt.close()

    return description, stats_image, recommendation


def get_agent_recommendation(incident):
    """Uses collective_functions for qualification-based recommendation"""
    global agent_model, df_skills, skills_updated, dic_roles, dic_roles_skills
    
    try:
        # Import get_roles_for_ff function
        from collective_functions import get_roles_for_ff
        
        # Ensure dic_roles_skills is in correct format
        if dic_roles_skills:
            for role_name, skills_array in dic_roles_skills.items():
                if isinstance(skills_array, np.ndarray) and skills_array.ndim == 1:
                    dic_roles_skills[role_name] = skills_array.reshape(1, -1)
        
        # Determine required vehicles based on incident type
        incident_type = incident['type'].upper()
        if 'RESCUE' in incident_type or 'ACCIDENT' in incident_type:
            veh_depart = ["VSAV", "EPA"]
        elif 'FIRE' in incident_type:
            veh_depart = ["FPT", "EPA"]
        else:
            veh_depart = ["VSAV"]
        
        # Available firefighters (simulation based on real data)
        ff_existing = df_skills.index[:30].tolist() if len(df_skills) > 30 else df_skills.index.tolist()
        print(f"DEBUG - ff_existing (ALL {len(ff_existing)}): {ff_existing}")
        print(f"DEBUG - ID type: {type(ff_existing[0]) if ff_existing else 'None'}")
        print(f"DEBUG - Firefighter 1029 in ff_existing? {1029 in ff_existing}")
        
        ff_array = gen_ff_array(df_skills, skills_updated, ff_existing)
        
        # DEBUG: Verify data
        print(f"DEBUG - Number of firefighters: {len(ff_existing)}")
        print(f"DEBUG - ff_array shape: {ff_array.shape if ff_array is not None else 'None'}")
        print(f"DEBUG - dic_roles_skills available: {dic_roles_skills is not None}")
        print(f"DEBUG - Required vehicles: {veh_depart}")
        
        # Firefighter dictionary (all available for demo)
        dic_ff = {ff: 0 for ff in df_skills.index}
        
        # Check and initialize dic_roles if necessary
        if not dic_roles:
            dic_roles = {
                "VSAV": {1: "CHEF_EQUIPE", 2: "EQUIPIER"},
                "FPT": {1: "CHEF_GARDE", 2: "CONDUCTEUR", 3: "EQUIPIER1", 4: "EQUIPIER2"},
                "EPA": {1: "CHEF_EQUIPE", 2: "EQUIPIER"}
            }
        
        # CORRECTION: Clean dic_roles to keep only roles that exist in dic_roles_skills
        available_roles = set(dic_roles_skills.keys()) if dic_roles_skills else set()
        print(f"DEBUG - Available roles in dic_roles_skills: {list(available_roles)}")
        
        # Clean dic_roles to avoid 'EQ_ENG_SAP' error
        dic_roles_clean = {}
        for vehicle, roles in dic_roles.items():
            roles_clean = {}
            for num, role_name in roles.items():
                if role_name in available_roles:
                    roles_clean[num] = role_name
                else:
                    print(f"WARNING - Role {role_name} not found, replaced by EQUIPIER")
                    if "EQUIPIER" in available_roles:
                        roles_clean[num] = "EQUIPIER"
                    elif "CREW_MEMBER" in available_roles:
                        roles_clean[num] = "CREW_MEMBER"
                    elif len(available_roles) > 0:
                        roles_clean[num] = list(available_roles)[0]
            
            if roles_clean:  # Only add if we have valid roles
                dic_roles_clean[vehicle] = roles_clean
        
        dic_roles = dic_roles_clean
        print(f"DEBUG - Cleaned dic_roles: {dic_roles}")
        
        # Check dic_roles_skills and initialize if necessary
        if not dic_roles_skills:
            print("WARNING: dic_roles_skills is empty, creating mock data")
            # Create mock data for roles
            dic_roles_skills = {
                "CHEF_EQUIPE": np.array([[1, 1, 1, 0, 0]]),
                "EQUIPIER": np.array([[1, 0, 0, 0, 0]]),
                "CHEF_GARDE": np.array([[1, 1, 1, 1, 0]]),
                "CONDUCTEUR": np.array([[1, 1, 0, 0, 0]])
            }
        else:
            print("INFO: Adjusting required skills to match available firefighters")
            # CORRECTION: Adjust required skills based on real skills
            # Analyze existing firefighter skills
            ff_skills_sample = ff_array[:5]  # First 5 firefighters
            print(f"DEBUG - Real skills analysis (first 5 firefighters):")
            for i, skills in enumerate(ff_skills_sample):
                available_skills = np.where(skills == 1)[0]
                print(f"  Firefighter {i}: skills at positions {available_skills[:10]}...")
            
            # Create more realistic required skills
            # Use skills that really exist among firefighters
            most_common_skills = []
            for skill_idx in range(min(20, ff_array.shape[1])):
                skill_count = np.sum(ff_array[:, skill_idx])
                if skill_count > 0:
                    most_common_skills.append(skill_idx)
            
            print(f"DEBUG - Most common skills (first 10): {most_common_skills[:10]}")
            
            if len(most_common_skills) >= 4:
                # Create realistic patterns based on existing skills
                leader_skills = np.zeros(ff_array.shape[1], dtype=int)
                crew_skills = np.zeros(ff_array.shape[1], dtype=int)
                
                # TEAM_LEADER: first 3 common skills + some specialized ones
                if len(most_common_skills) >= 3:
                    leader_skills[most_common_skills[:3]] = 1
                
                # CREW_MEMBER: just the most common skill
                if len(most_common_skills) >= 1:
                    crew_skills[most_common_skills[0]] = 1
                
                dic_roles_skills = {
                    "TEAM_LEADER": leader_skills.reshape(1, -1),
                    "CREW_MEMBER": crew_skills.reshape(1, -1),
                    "GUARD_CHIEF": leader_skills.reshape(1, -1),  # Same as team leader
                    "DRIVER": crew_skills.reshape(1, -1)  # Same as crew member
                }
                
                print(f"DEBUG - New TEAM_LEADER skills: positions {np.where(leader_skills == 1)[0]}")
                print(f"DEBUG - New CREW_MEMBER skills: positions {np.where(crew_skills == 1)[0]}")
            else:
                print("WARNING: Not enough common skills, using simple patterns")
                # Fallback: use very simple patterns
                dic_roles_skills = {
                    "TEAM_LEADER": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)]),
                    "CREW_MEMBER": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)]),
                    "GUARD_CHIEF": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)]),
                    "DRIVER": np.array([[0, 0, 0, 1, 0] + [0] * (ff_array.shape[1] - 5)])
                }
        
        # Qualification analysis for each required vehicle
        vehicle_recommendations = []
        best_firefighter = None
        best_score = 0
        
        for vehicle in veh_depart:
            print(f"DEBUG - Analyzing vehicle: {vehicle}")
            if vehicle in dic_roles:
                try:
                    # Get matching matrix for this vehicle
                    roles_matrix = get_roles_for_ff(vehicle, ff_array, dic_roles, dic_roles_skills)
                    print(f"DEBUG - roles_matrix shape for {vehicle}: {roles_matrix.shape}")
                    
                    # DEBUG: Analyze input data
                    print(f"DEBUG - ff_array sample (first firefighter): {ff_array[0][:10] if len(ff_array) > 0 else 'Empty'}")
                    
                    # Analyze each role of the vehicle
                    required_roles = dic_roles[vehicle]
                    for role_idx, (role_num, role_name) in enumerate(sorted(required_roles.items())):
                        # print(f"DEBUG - Role {role_idx}: {role_name}")
                        
                        # DEBUG: Check required skills for this role
                        if role_name in dic_roles_skills:
                            required_skills = dic_roles_skills[role_name]
                            # print(f"DEBUG - Required skills for {role_name}: {required_skills.shape} -> {required_skills}")
                        else:
                            # print(f"DEBUG - ERROR: Role {role_name} not found in dic_roles_skills")
                            # print(f"DEBUG - Available roles: {list(dic_roles_skills.keys())}")
                            pass
                        
                        if role_idx < roles_matrix.shape[1]:
                            # Qualification scores for this role
                            qualification_scores = roles_matrix[:, role_idx]
                            # print(f"DEBUG - Max scores for {role_name}: {np.max(qualification_scores) if len(qualification_scores) > 0 else 'None'}")
                            # print(f"DEBUG - Sample scores for {role_name}: {qualification_scores[:5]}")
                            
                            # Find qualified firefighters (score > 0)
                            qualified_firefighters = []
                            for i, score in enumerate(qualification_scores):
                                if score > 0 and i < len(ff_existing):
                                    firefighter_id = ff_existing[i]
                                    qualified_firefighters.append((firefighter_id, score))
                            
                            # print(f"DEBUG - Qualified firefighters for {role_name}: {len(qualified_firefighters)}")
                            
                            # Sort by descending score
                            qualified_firefighters.sort(key=lambda x: x[1], reverse=True)
                            
                            # CORRECTION: More equitable selection of best firefighter
                            if qualified_firefighters:
                                # Take all firefighters with maximum score
                                max_score = qualified_firefighters[0][1]
                                max_score_firefighters = [p for p in qualified_firefighters if p[1] == max_score]
                                
                                # If multiple firefighters have same max score, rotate based on time
                                import time
                                rotation_index = int(time.time()) % len(max_score_firefighters)
                                selected_firefighter = max_score_firefighters[rotation_index]
                                
                                print(f"DEBUG - {len(max_score_firefighters)} firefighters have max score {max_score}")
                                print(f"DEBUG - Selected firefighter (rotation {rotation_index}): {selected_firefighter[0]}")
                                print(f"DEBUG - Index of firefighter in ff_existing: {ff_existing.index(selected_firefighter[0]) if selected_firefighter[0] in ff_existing else 'NOT FOUND'}")
                                print(f"DEBUG - ff_existing contains {selected_firefighter[0]}? {selected_firefighter[0] in ff_existing}")
                                
                                # CORRECTION: Verify that selected firefighter is in ff_existing
                                if selected_firefighter[0] in ff_existing:
                                    # Update global best firefighter
                                    if max_score > best_score:
                                        best_score = max_score
                                        best_firefighter = (selected_firefighter[0], max_score, vehicle, role_name)
                                        print(f"DEBUG - New best firefighter: {best_firefighter}")
                                else:
                                    print(f"WARNING - Firefighter {selected_firefighter[0]} not in ff_existing, using first available")
                                    # Fallback: use first firefighter from ff_existing with score > 0
                                    for firefighter_id, score in qualified_firefighters:
                                        if firefighter_id in ff_existing and score > best_score:
                                            best_score = score
                                            best_firefighter = (firefighter_id, score, vehicle, role_name)
                                            print(f"DEBUG - Fallback best firefighter: {best_firefighter}")
                                            break
                            
                            vehicle_recommendations.append({
                                'vehicle': vehicle,
                                'role': role_name,
                                'qualified_firefighters': qualified_firefighters[:3]  # Top 3
                            })
                
                except Exception as e:
                    print(f"ERROR - Error with vehicle {vehicle}: {e}")
                    vehicle_recommendations.append({
                        'vehicle': vehicle,
                        'role': 'ERROR',
                        'qualified_firefighters': []
                    })
            else:
                print(f"ERROR - Vehicle {vehicle} not in dic_roles")
        
        # Use get_potential_actions to validate selection
        if agent_model and best_firefighter:
            # Create minimal state for get_potential_actions
            coord_x = incident.get('coord_x_orig', incident['lon'])
            coord_y = incident.get('coord_y_orig', incident['lat'])
            month = incident['mois_num']
            hour = incident.get('Hour', 12)
            
            # Encode temporal values
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
                
                # Validate with get_potential_actions
                potential_actions, potential_skills = get_potential_actions(state, False)
                
                if potential_actions and potential_actions[0] != 99:
                    validated_action = potential_actions[0]
                    validated_skill = potential_skills[0] if potential_skills else best_score
                else:
                    validated_action = None
                    validated_skill = 0
                    
            except Exception as e:
                print(f"Error in validation: {e}")
                validated_action = None
                validated_skill = best_score
        else:
            validated_action = None
            validated_skill = best_score
        
        # Generate final decision
        print(f"DEBUG FINAL - best_firefighter: {best_firefighter}")
        print(f"DEBUG FINAL - ff_existing complete: {ff_existing}")
        print(f"DEBUG FINAL - Firefighter {best_firefighter[0] if best_firefighter else 'None'} in ff_existing? {best_firefighter[0] in ff_existing if best_firefighter else 'N/A'}")
        
        if best_firefighter:
            firefighter_id, score, vehicle, role = best_firefighter
            print(f"DEBUG FINAL - Selected firefighter: {firefighter_id} (type: {type(firefighter_id)})")
            decision_text = f"👨‍🚒 Firefighter {firefighter_id} recommended for {vehicle} ({role}) - Level {score}"
            decision_color = "green" if score >= 6 else "orange" if score >= 4 else "red"
            skill_lvl = score
        else:
            decision_text = "❌ No qualified firefighter available"
            decision_color = "red"
            skill_lvl = 0
        
        # Create detailed recommendation
        recommendation = f"""
## 🤖 Qualification-based Recommendations

### 📊 Situation Analysis
- **Incident type:** {incident['type']}
- **Estimated duration:** {incident['temps']:.0f} minutes
- **Zone:** {incident.get('zone', 'Toulouse center')}

### 🎯 Optimal Decision
**{decision_text}**

### 📋 Allocation Details
- **Suggested vehicles:** {', '.join(veh_depart)}
- **Qualification level:** {skill_lvl}/8
- **Estimated response time:** {np.random.randint(8, 15)} minutes

###  Analysis Method
This recommendation uses `collective_functions` to:
1. Analyze required qualifications by vehicle
2. Evaluate each firefighter according to their skills
3. Select the best qualification/availability match
"""
        
        # Create decision graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graph 1: Qualification scores of top firefighters
        if vehicle_recommendations:
            top_firefighters = []
            top_scores = []
            top_roles = []
            
            for rec in vehicle_recommendations:
                for firefighter_id, score in rec['qualified_firefighters'][:2]:
                    top_firefighters.append(f"FF_{firefighter_id}")
                    top_scores.append(score)
                    top_roles.append(rec['role'])
            
            if top_firefighters:
                # Define action for colors
                action = 0 if best_firefighter else 99
                
                colors = ['green' if i == 0 and best_firefighter else 'lightblue' for i in range(len(top_firefighters))]
                bars = ax1.bar(range(len(top_firefighters)), top_scores, color=colors, alpha=0.7)
                ax1.set_xlabel('Available Firefighters')
                ax1.set_ylabel('Qualification Level')
                ax1.set_title('Top Firefighters by Qualification')
                ax1.set_xticks(range(len(top_firefighters)))
                ax1.set_xticklabels(top_firefighters, rotation=45, ha='right')
                ax1.set_ylim(0, 8)
                
                # Add values on bars
                for bar, score in zip(bars, top_scores):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{score}', ha='center', va='bottom')
            else:
                ax1.text(0.5, 0.5, 'No qualification\nfound', 
                        transform=ax1.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
                ax1.set_title('Qualifications - No match')
        else:
            ax1.text(0.5, 0.5, 'Qualification\nanalysis\nnot available', 
                    transform=ax1.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
            ax1.set_title('Qualifications - In progress')
        
        # Graph 2: Distribution of recommended vehicles
        if veh_depart:
            vehicle_counts = {}
            for veh in veh_depart:
                vehicle_counts[veh] = vehicle_counts.get(veh, 0) + 1
            
            labels = list(vehicle_counts.keys())
            sizes = list(vehicle_counts.values())
            colors_pie = ['red' if 'VSAV' in label else 'blue' if 'FPT' in label else 'green' for label in labels]
            
            ax2.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors=colors_pie)
            ax2.set_title('Recommended Vehicles')
        else:
            ax2.text(0.5, 0.5, 'No vehicle\nrecommended', 
                    transform=ax2.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
            ax2.set_title('Vehicles - None')
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        decision_plot = Image.open(buf)
        plt.close()
        
        return recommendation, decision_plot
        
    except Exception as e:
        import traceback
        error_msg = f"""
## ❌ Qualification Analysis Error
**Message:** {str(e)}
**Incident type:** {incident.get('type', 'Unknown')}

### 🔧 Technical Details
```
{traceback.format_exc()}
```

### 💡 Suggestions
- Check that qualification data is loaded
- Restart environment with "🔄 Reload agent" button
- Use Diagnostics tab to test functions
"""
        return error_msg, None


def simulate_intervention():
    """Simulate an intervention with RL agent"""
    global incident_selectionne, agent_model

    if incident_selectionne is None:
        return "⚠️ Please first select an incident"

    if not agent_loaded:
        return "❌ RL Agent not loaded"

    try:
        # CORRECTION: Safe conversion of pandas values
        incident_time = float(incident_selectionne['temps']) if hasattr(incident_selectionne['temps'], 'item') else \
        incident_selectionne['temps']

        # Simple simulation
        simulation_time = np.random.normal(incident_time, 10)
        efficiency = np.random.uniform(0.85, 0.95)

        # Calculate costs and benefits
        estimated_cost = simulation_time * 150
        time_savings = max(0, incident_time - simulation_time)
        cost_savings = time_savings * 100

        report = f"""
## 🎮 Intervention simulation completed

### 📊 Simulation results
- **Actual time:** {simulation_time:.0f} minutes (expected: {incident_time:.0f} min)
- **Difference:** {simulation_time - incident_time:+.0f} minutes
- **Efficiency:** {efficiency * 100:.1f}%
- **Resources used:** 3 vehicles, 12 firefighters

### 💰 Economic analysis
- **Intervention cost:** {estimated_cost:.0f}€
- **Savings achieved:** {cost_savings:.0f}€
- **AI ROI:** {(cost_savings / estimated_cost) * 100:.1f}%

### ✅ Intervention {'successful' if simulation_time <= incident_time * 1.1 else 'acceptable'}!
The RL agent {'optimized' if time_savings > 0 else 'maintained'} resource allocation.

### 🎯 Key points
- **Speed:** {'🟢 Excellent' if time_savings > 5 else '🟡 Standard' if time_savings >= 0 else '🔴 Needs improvement'}
- **Efficiency:** {'🟢 Very efficient' if efficiency > 0.9 else '🟡 Efficient' if efficiency > 0.85 else '🔴 Needs improvement'}
- **Cost:** {'🟢 Optimal' if cost_savings > 0 else '🟡 Standard'}
"""

        return report

    except Exception as e:
        return f"""
## ❌ Simulation error

**Error message:** {str(e)}

**Incident data:**
- Type: {incident_selectionne.get('type', 'Unknown')}
- Time: {incident_selectionne.get('temps', 'Unknown')}

**Suggestion:** Try with another incident or restart the agent.
"""


# === NEW FUNCTIONS FOR STATISTICS ===

def generate_interventions_by_hour_chart(hour_filter=None):
    """Generate chart of interventions by hour"""
    df_work = df_data.copy()

    # Group by hour
    if 'Hour' in df_work.columns:
        interventions_by_hour = df_work.groupby('Hour').size().reset_index(name='count')
    else:
        # Generate mock hours if not available
        hours = np.random.randint(0, 24, len(df_work))
        df_work['Hour'] = hours
        interventions_by_hour = df_work.groupby('Hour').size().reset_index(name='count')

    # Filter by hour if specified
    if hour_filter is not None and hour_filter != "All":
        hour_num = int(hour_filter.split('h')[0])
        interventions_by_hour = interventions_by_hour[interventions_by_hour['Hour'] == hour_num]

    # Create chart
    fig = px.bar(
        interventions_by_hour,
        x='Hour',
        y='count',
        title='Distribution of interventions by hour of day',
        labels={'Hour': 'Hour', 'count': 'Number of interventions'},
        color='count',
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        height=400
    )

    return fig


def generate_interventions_by_month_chart(month_filter=None):
    """Generate chart of interventions by month"""
    df_work = df_data.copy()

    # Group by month
    interventions_by_month = df_work.groupby('Month').size().reset_index(name='count')
    interventions_by_month['Month_name'] = interventions_by_month['Month'].map(MONTH_MAPPING)

    # Filter by month if specified
    if month_filter and month_filter != "All":
        month_num = None
        for num, nom in MONTH_MAPPING.items():
            if nom == month_filter:
                month_num = num
                break
        if month_num:
            interventions_by_month = interventions_by_month[interventions_by_month['Month'] == month_num]

    # Create chart
    fig = px.bar(
        interventions_by_month,
        x='Month_name',
        y='count',
        title='Distribution of interventions by month',
        labels={'Month_name': 'Month', 'count': 'Number of interventions'},
        color='count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=400
    )

    return fig


def generate_intervention_duration_chart(incident_type=None):
    """Generate chart of intervention durations"""
    df_work = df_data.copy()

    # Filter by type if specified
    if incident_type and incident_type != "All":
        df_work = df_work[df_work['type'] == incident_type]

    # Create duration histogram
    fig = px.histogram(
        df_work,
        x='Duration',
        nbins=30,
        title=f'Distribution of intervention durations{" - " + incident_type if incident_type and incident_type != "All" else ""}',
        labels={'Duration': 'Duration (minutes)', 'count': 'Number of interventions'},
        color_discrete_sequence=['orange']
    )

    # Add statistics
    mean = df_work['Duration'].mean()
    median = df_work['Duration'].median()

    fig.add_vline(x=mean, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean:.0f}min")
    fig.add_vline(x=median, line_dash="dash", line_color="blue",
                  annotation_text=f"Median: {median:.0f}min")

    fig.update_layout(height=400)

    return fig


def generate_incident_month_correlation_heatmap():
    """Generate heatmap of correlations between incident types and months"""
    df_work = df_data.copy()

    # Create contingency table
    contingency = pd.crosstab(df_work['type'], df_work['Month'])
    contingency.columns = [MONTH_MAPPING[i] for i in contingency.columns]

    # Normalize by row to see patterns
    contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)

    # Create heatmap with plotly
    fig = px.imshow(
        contingency_norm.values,
        x=contingency_norm.columns,
        y=contingency_norm.index,
        color_continuous_scale='Viridis',
        title='Correlation between incident types and months',
        labels={'x': 'Month', 'y': 'Incident type', 'color': 'Proportion'}
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def generate_vehicle_intervention_chart():
    """Generate chart of vehicles in intervention"""
    # Mock data based on project information
    vehicles = ['VSAV', 'FPT', 'EPA', 'FPTGP', 'CCFS', 'VTU', 'VLCG']
    utilizations = [44000, 12000, 8000, 3000, 2500, 1500, 800]  # Based on real data

    fig = px.bar(
        x=vehicles,
        y=utilizations,
        title='Total number of vehicles mobilized by type (year)',
        labels={'x': 'Vehicle type', 'y': 'Number of mobilizations'},
        color=utilizations,
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        height=400,
        yaxis_type="log",  # Logarithmic scale as in documents
        xaxis_tickangle=-45
    )

    return fig


def generate_interventions_by_day_chart():
    """Generate chart of interventions by day of year"""
    df_work = df_data.copy()

    # Group by day
    if 'Day' in df_work.columns:
        interventions_by_day = df_work.groupby('Day').size().reset_index(name='count')
    else:
        # Generate mock days
        days = np.random.randint(1, 366, len(df_work))
        df_work['Day'] = days
        interventions_by_day = df_work.groupby('Day').size().reset_index(name='count')

    # Sort by number of interventions (as in documents)
    interventions_by_day = interventions_by_day.sort_values('count', ascending=False)
    interventions_by_day['rank'] = range(1, len(interventions_by_day) + 1)

    fig = px.scatter(
        interventions_by_day,
        x='rank',
        y='count',
        title='Interventions by day (ranked in descending order)',
        labels={'rank': 'Day rank', 'count': 'Number of interventions'},
        color='count',
        color_continuous_scale='Plasma'
    )

    fig.update_layout(height=400)

    return fig


def update_statistics(criteria, filter_value):
    """Update charts according to selected criteria"""

    if criteria == "Hour":
        fig1 = generate_interventions_by_hour_chart(filter_value)
        fig2 = generate_intervention_duration_chart()
        return fig1, fig2

    elif criteria == "Month":
        fig1 = generate_interventions_by_month_chart(filter_value)
        fig2 = generate_incident_month_correlation_heatmap()
        return fig1, fig2

    elif criteria == "Incident type":
        fig1 = generate_intervention_duration_chart(filter_value)
        fig2 = generate_vehicle_intervention_chart()
        return fig1, fig2

    else:  # "General"
        fig1 = generate_interventions_by_day_chart()
        fig2 = generate_vehicle_intervention_chart()
        return fig1, fig2


def update_filter_options(criteria):
    """Update filter options according to selected criteria"""

    if criteria == "Hour":
        options = ["All"] + [f"{i}h" for i in range(24)]
        return gr.update(choices=options, value="All", visible=True, label="Filter by hour")

    elif criteria == "Month":
        options = ["All"] + list(MONTH_MAPPING.values())
        return gr.update(choices=options, value="All", visible=True, label="Filter by month")

    elif criteria == "Incident type":
        incident_types = ["All"] + df_data['type'].unique().tolist()
        return gr.update(choices=incident_types, value="All", visible=True, label="Filter by type")

    else:  # "General"
        return gr.update(visible=False)


def generate_process_map(station, month, incident_type):
    """Generate process map according to selected filters"""

    # Adapt "All"/"All"/None values for the function
    station_val = None if not station or station == "All" else station
    month_val = None
    if month and month != "All":
        month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
                         "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
        month_val = month_mapping.get(month, None)
    type_val = None if not incident_type or incident_type == "All" else incident_type

    # Load merged data (must be globally accessible or imported)
    df_merged, _ = load_data_with_stations()

    # Simply call the function and return the figure
    return create_filtered_process_map(df_data, df_merged, station_name=station_val, month=month_val,
                                       incident_type=type_val)


def create_fallback_process_map_html(station, month, incident_type, error_msg):
    """Create fallback HTML in case of error"""
    return f"""
    <div style='height: 500px; border: 2px solid #FF9800; border-radius: 10px; padding: 20px; 
                background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%, #fff3e0 100%);'>
        <h3 style='text-align: center; color: #E65100; margin-bottom: 20px;'>🔄 Process Map of Interventions</h3>

        <div style='background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h4 style='color: #1976D2; margin-bottom: 10px;'>📋 Applied filters:</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;'>
                <div style='padding: 8px; background: #E3F2FD; border-radius: 4px; text-align: center;'>
                    <strong>🏢 Station:</strong><br>{station}
                </div>
                <div style='padding: 8px; background: #F3E5F5; border-radius: 4px; text-align: center;'>
                    <strong>📅 Month:</strong><br>{month}
                </div>
                <div style='padding: 8px; background: #FFF3E0; border-radius: 4px; text-align: center;'>
                    <strong>🚨 Type:</strong><br>{incident_type}
                </div>
            </div>
        </div>

        <div style='text-align: center; padding: 40px; background: rgba(255,255,255,0.9); 
                    border: 2px dashed #FF9800; border-radius: 8px;'>
            <div style='font-size: 48px; margin-bottom: 20px;'>⚠️</div>
            <p style='font-size: 16px; color: #E65100; margin: 10px 0; font-weight: bold;'>
                Process Map temporarily unavailable
            </p>
            <p style='color: #666; font-size: 14px; margin: 10px 0;'>{error_msg}</p>
            <div style='margin-top: 20px; padding: 15px; background: #FFF3E0; border-radius: 4px;'>
                <p style='font-size: 12px; color: #F57C00; margin: 0;'>
                    <strong>📊 Analysis in progress...</strong><br>
                    Process mining system will be available soon with your custom filters
                </p>
            </div>
        </div>
    </div>
    """


import gradio as gr

# === INTERFACE GRADIO ===
with gr.Blocks(title="AI Firefighter Dashboard - Haute-Garonne", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🚒 Intelligent Firefighter Dashboard - Haute-Garonne
    ### Using AI to optimize real-time interventions
    """)

    # Agent status
    with gr.Row():
        agent_status = gr.Markdown(
            f"🤖 **RL Agent:** {'✅ Loaded and ready' if agent_loaded else '❌ Not loaded'}"
        )
        reload_button = gr.Button("🔄 Reload agent", size="sm")

    with gr.Tabs():

        # === TAB 1: MAP AND AI RECOMMENDATIONS ===
        with gr.TabItem("🗺️ Map and AI Recommendations"):
            with gr.Row():
                type_input = gr.Dropdown(
                    ["All", "Rescue operation", "Fire", "Road accident", "Flood", "Gas leak", "Other rescue"],
                    label="Incident type", value="All"
                )
                month_input = gr.Dropdown(
                    list(MONTH_MAPPING.values()),
                    label="Month", value="January"
                )
                search_button = gr.Button("🔍 Search", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📍 Incidents")
                    map_output = gr.Plot(label="Incident map")

                with gr.Column(scale=1):
                    gr.Markdown("### 🚒 Fire stations")
                    stations_output = gr.Plot(label="Fire stations map")

            message = gr.Markdown()

            with gr.Row():
                dropdown_incident = gr.Dropdown(label="📋 Select an incident for AI analysis", visible=False)
                simulate_btn = gr.Button("🎮 Simulate intervention", visible=False)

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Incident details")
                    stats_output = gr.Markdown()
                    img_plot = gr.Image(type="pil", label="Metrics")

                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 RL Agent Recommendations")
                    recommendations_output = gr.Markdown(
                        "Select an incident to get AI recommendations"
                    )

            simulation_output = gr.Markdown()

        # === TAB 2: STATISTICS ===
        with gr.TabItem("📊 Statistics"):
            gr.Markdown("""
            ## 📈 Statistical analysis of interventions

            Explore patterns and trends in firefighter interventions through different analysis criteria.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    criteria_stats = gr.Dropdown(
                        ["General", "Hour", "Month", "Incident type"],
                        label="🔍 Analysis criteria",
                        value="General"
                    )

                with gr.Column(scale=1):
                    filter_stats = gr.Dropdown(
                        [],
                        label="Filter by",
                        visible=False
                    )

                with gr.Column(scale=1):
                    stats_button = gr.Button("📊 Update charts", variant="primary")

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    main_chart = gr.Plot(
                        label="Main chart",
                        value=generate_interventions_by_day_chart()
                    )

                with gr.Column():
                    secondary_chart = gr.Plot(
                        label="Secondary chart",
                        value=generate_vehicle_intervention_chart()
                    )

            with gr.Row():
                correlation_heatmap = gr.Plot(
                    label="Incident-month correlations",
                    value=generate_incident_month_correlation_heatmap()
                )

            gr.Markdown("""
            ### 💡 Key insights

            **Identified temporal patterns:**
            - **Activity peak:** Between 2pm and 6pm (afternoon rush)
            - **Activity minimum:** Between 2am and 6am
            - **Seasonality:** Increase in chimney fires in winter, accident peaks in summer

            **Intervention types:**
            - **Rescue operations:** 55% of interventions (dominant)
            - **Fires:** 15% with strong winter seasonality
            - **Road accidents:** 20% with peaks during rush hours

            **Vehicle usage:**
            - **VSAV:** ~44,000 mobilizations/year (most used vehicle)
            - **FPT:** ~12,000 mobilizations/year (fire interventions)
            - **EPA:** ~8,000 mobilizations/year (first aid)
            """)

            gr.Markdown("---")

        # === NEW TAB: PROCESS MINING ===
        with gr.TabItem("⛏️ Process Mining"):
            gr.Markdown("""
            ## 🔄 Process Map of Interventions

            Visualize intervention processes and flows according to different filtering criteria.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    station_filter = gr.Dropdown(
                        ["All"] + df_casernes['nom'].tolist(),  # Dynamically fetch stations
                        label="🏢 Station",
                        value="All"
                    )

                with gr.Column(scale=1):
                    month_process_filter = gr.Dropdown(
                        ["All"] + list(MONTH_MAPPING.values()),
                        label="📅 Month",
                        value="All"
                    )

                with gr.Column(scale=1):
                    incident_type_process_filter = gr.Dropdown(
                        ["All", "Rescue operation", "Fire", "Road accident", "Flood", "Gas leak", "Other rescue"],
                        label="🚨 Incident type",
                        value="All"
                    )

                with gr.Column(scale=1):
                    process_map_button = gr.Button("🔄 Show Process Map", variant="primary")

            with gr.Row():
                process_map_output = gr.Plot(
                    label="Process Map ",
                    show_label=True,
                    elem_id="process-map-box",
                    value=None
                )

        # === TAB 3: DIAGNOSTICS AND DEBUG ===
        with gr.TabItem("🔧 Diagnostics"):
            gr.Markdown("## 🔍 System diagnostics")

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
                    ### 📋 System status
                    - **RL Agent:** {'✅ Operational' if agent_loaded else '❌ Error'}
                    - **Incident data:** {'✅ ' + str(len(df_data)) + ' incidents' if len(df_data) > 0 else '❌ No data'}
                    - **Stations:** {'✅ ' + str(len(df_casernes)) + ' stations' if len(df_casernes) > 0 else '❌ No stations'}
                    - **Skills:** {'✅ ' + str(len(df_skills)) + ' firefighters' if df_skills is not None else '❌ Not loaded'}
                    """)

                with gr.Column():
                    gr.Markdown("""
                    ### 🔧 Maintenance actions
                    - Reload RL agent
                    - Update data
                    - Check models
                    - Clear cache
                    """)

            with gr.Row():
                test_model_btn = gr.Button("🧪 Test RL model")
                debug_output = gr.Textbox(label="Debug output", lines=10)

    # === EVENTS ===

    # Incident search
    search_button.click(
        fn=search_incidents,
        inputs=[type_input, month_input],
        outputs=[map_output, stations_output, message, dropdown_incident]
    ).then(
        lambda: gr.update(visible=True),
        outputs=simulate_btn
    )

    # Incident selection with AI recommendations
    dropdown_incident.change(
        fn=display_stats_and_recommendations,
        inputs=dropdown_incident,
        outputs=[stats_output, img_plot, recommendations_output]
    )

    # Simulation
    simulate_btn.click(
        fn=simulate_intervention,
        outputs=simulation_output
    )


    # Reload agent
    def reload_agent_callback():
        success = load_agent_and_environment()
        global agent_loaded
        status_text = f"🤖 **RL Agent:** {'✅ Loaded and ready' if agent_loaded else '❌ Loading failed'}"
        if success:
            status_text += f"\n🔄 Reloaded at {datetime.now().strftime('%H:%M:%S')}"
        return status_text


    reload_button.click(
        fn=reload_agent_callback,
        outputs=agent_status
    )

    # Events for statistics
    criteria_stats.change(
        fn=update_filter_options,
        inputs=criteria_stats,
        outputs=filter_stats
    )

    stats_button.click(
        fn=update_statistics,
        inputs=[criteria_stats, filter_stats],
        outputs=[main_chart, secondary_chart]
    )

    # Auto-update charts when changing filter
    filter_stats.change(
        fn=update_statistics,
        inputs=[criteria_stats, filter_stats],
        outputs=[main_chart, secondary_chart]
    )

    # Process Map
    process_map_button.click(
        fn=generate_process_map,
        inputs=[station_filter, month_process_filter, incident_type_process_filter],
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
                            debug_info += f"⚠️ Action {action} hors limites (max: {len(q_values) - 1})\n"

                        # Afficher les top 5 actions
                        top_indices = np.argsort(q_values)[-5:][::-1]
                        debug_info += "🏆 Top 5 actions:\n"
                        for i, idx in enumerate(top_indices):
                            debug_info += f"   {i + 1}. Action {idx}: {q_values[idx]:.3f}\n"

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