import pandas as pd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np
from datetime import datetime
import seaborn as sns
import re

def load_data_with_stations(csv_file='event_logs_with_vehicles.csv'):
    """
    Charger les données modifiées avec types de véhicules et informations des casernes
    """
    print("📂 Chargement des données modifiées...")
    
    # Charger df_pc_real qui contient les casernes
    with open('df_pc_real.pkl', 'rb') as f:
        df_real = pickle.load(f)
    
    # Charger les logs d'événements MODIFIÉS
    df_events = pd.read_csv(csv_file)
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

def analyze_vehicle_activities(df_merged):
    """
    Analyser les activités de véhicules modifiées
    """
    print("\n🚗 Analyse des activités de véhicules...")
    
    # Filtrer les activités de départ
    vehicle_activities = df_merged[df_merged['activity'].str.contains('Départ_véhicule_', na=False)]
    
    if len(vehicle_activities) > 0:
        print(f"\nNombre d'activités de départ: {len(vehicle_activities)}")
        
        # Analyser les patterns
        activity_counts = vehicle_activities['activity'].value_counts()
        print("\nTop 15 des départs de véhicules:")
        for activity, count in activity_counts.head(15).items():
            print(f"  - {activity}: {count} fois")
        
        # Extraire les statistiques par position
        position_stats = {}
        for activity in vehicle_activities['activity'].unique():
            match = re.search(r'Départ_véhicule_(\d+)_(.+)', activity)
            if match:
                position = int(match.group(1))
                vehicle_type = match.group(2)
                
                if position not in position_stats:
                    position_stats[position] = {}
                if vehicle_type not in position_stats[position]:
                    position_stats[position][vehicle_type] = 0
                position_stats[position][vehicle_type] += 1
        
        print("\n📊 Types de véhicules par position:")
        for pos in sorted(position_stats.keys()):
            print(f"\n🚒 Position {pos}:")
            sorted_types = sorted(position_stats[pos].items(), 
                                key=lambda x: x[1], reverse=True)
            for v_type, count in sorted_types[:5]:
                print(f"  - {v_type}: {count} fois")
    
    return vehicle_activities

def calculate_transition_times_enhanced(df_filtered):
    """
    Calculer les temps moyens entre transitions avec analyse des véhicules
    """
    transition_times = {}
    transition_counts = {}
    vehicle_transition_times = {}  # Nouveau : temps par type de véhicule
    
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
            
            # Compter les transitions
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
            
            # Analyser spécifiquement les transitions de véhicules
            if 'Départ_véhicule_' in current['activity']:
                match = re.search(r'Départ_véhicule_\d+_(.+)', current['activity'])
                if match:
                    vehicle_type = match.group(1)
                    key = f"{vehicle_type} → {next_act['activity']}"
                    
                    if key not in vehicle_transition_times:
                        vehicle_transition_times[key] = []
                    vehicle_transition_times[key].append(time_diff)
    
    # Calculer les moyennes
    avg_times = {}
    for transition, times in transition_times.items():
        avg_times[transition] = {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'count': len(times)
        }
    
    # Moyennes pour les transitions de véhicules
    avg_vehicle_times = {}
    for transition, times in vehicle_transition_times.items():
        if len(times) >= 5:  # Au moins 5 occurrences
            avg_vehicle_times[transition] = {
                'mean': np.mean(times),
                'median': np.median(times),
                'count': len(times)
            }
    
    return avg_times, transition_counts, avg_vehicle_times

def create_enhanced_process_map(station_name, df_merged, save_path='process_map_enhanced.png'):
    """
    Créer un process map amélioré avec les types de véhicules
    """
    print(f"\n🗺️ Création du Process Map amélioré pour: {station_name}")
    
    # Filtrer par caserne
    df_station = df_merged[df_merged['station'] == station_name]
    
    if len(df_station) == 0:
        print(f"❌ Aucune donnée trouvée pour la caserne: {station_name}")
        return
    
    n_cases = df_station['case_id'].nunique()
    print(f"✅ {n_cases} interventions trouvées pour {station_name}")
    
    # Calculer les temps de transition
    avg_times, transition_counts, avg_vehicle_times = calculate_transition_times_enhanced(df_station)
    
    # OPTIMISATION : Filtrer les activités peu fréquentes
    min_activity_count = max(5, n_cases * 0.01)  # Au moins 1% des cas
    activity_counts = df_station['activity'].value_counts()
    frequent_activities = activity_counts[activity_counts >= min_activity_count].index.tolist()
    
    print(f"📊 Activités fréquentes retenues: {len(frequent_activities)}/{len(activity_counts)}")
    
    # Créer le graphe avec seulement les activités fréquentes
    G = nx.DiGraph()
    
    # Ajouter les nœuds fréquents
    for activity in frequent_activities:
        count = activity_counts[activity]
        G.add_node(activity, count=count)
    
    # Ajouter les arêtes entre activités fréquentes
    for (from_act, to_act), stats in avg_times.items():
        if from_act in frequent_activities and to_act in frequent_activities and stats['count'] >= 5:
            G.add_edge(from_act, to_act, 
                      weight=stats['mean'],
                      count=stats['count'],
                      median=stats['median'])
    
    # OPTIMISATION : Réduire la taille de la figure
    n_nodes = len(G.nodes())
    fig_width = min(20, max(12, n_nodes * 0.8))
    fig_height = min(14, max(8, n_nodes * 0.5))
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Layout hiérarchique optimisé
    pos = {}
    
    # Positions fixes pour les activités principales
    fixed_positions = {
        'Appel_reçu': (0, 0),
        'Décision_départ': (2, 0),
        'Arrivée_sur_site': (10, 0),
        'Intervention_en_cours': (12, 0),
        'Fin_intervention': (14, 0)
    }
    
    # OPTIMISATION : Regrouper les véhicules similaires
    vehicle_groups = {}
    vehicle_activities = [act for act in G.nodes() if 'Départ_véhicule_' in act]
    
    for activity in vehicle_activities:
        match = re.search(r'Départ_véhicule_(\d+)_(.+)', activity)
        if match:
            vehicle_num = int(match.group(1))
            vehicle_type = match.group(2).split('_')[0]  # Prendre seulement le premier type
            
            key = f"V{vehicle_num}_{vehicle_type}"
            if key not in vehicle_groups:
                vehicle_groups[key] = []
            vehicle_groups[key].append(activity)
    
    # Positionner les groupes de véhicules
    x_offset = 4
    y_offset = 3
    max_per_column = 8  # Maximum d'activités par colonne
    
    col = 0
    row = 0
    for group_key in sorted(vehicle_groups.keys()):
        activities = vehicle_groups[group_key][:max_per_column]  # Limiter le nombre
        
        for i, activity in enumerate(activities):
            x = x_offset + col * 2
            y = y_offset - row * 0.8
            pos[activity] = (x, y)
            
            row += 1
            if row >= max_per_column:
                row = 0
                col += 1
    
    # Ajouter les positions fixes
    for node, position in fixed_positions.items():
        if node in G.nodes():
            pos[node] = position
    
    # Pour les nœuds manquants
    missing_nodes = [node for node in G.nodes() if node not in pos]
    if missing_nodes:
        # Utiliser un layout simple pour les nœuds manquants
        y_offset_missing = -3
        for i, node in enumerate(missing_nodes):
            pos[node] = (6 + (i % 3) * 2, y_offset_missing - (i // 3) * 0.8)
    
    # Dessiner les nœuds
    for node in G.nodes():
        if node not in pos:
            continue
            
        x, y = pos[node]
        count = G.nodes[node]['count']
        
        # Taille proportionnelle mais limitée
        size = min(1.5, 0.5 + (count / n_cases) * 0.8)
        
        # Couleur selon le type
        if 'Départ_véhicule_' in node:
            if 'VSAV' in node:
                color = 'lightcoral'
            elif 'FPT' in node:
                color = 'lightgreen'
            elif 'EPA' in node:
                color = 'lightyellow'
            elif 'CCF' in node:
                color = 'lightpink'
            else:
                color = 'lightgray'
        elif node in ['Appel_reçu', 'Décision_départ']:
            color = 'lightblue'
        elif node == 'Fin_intervention':
            color = 'lightsteelblue'
        else:
            color = 'wheat'
        
        # Créer une boîte arrondie
        box = FancyBboxPatch((x-size/2, y-0.25), size, 0.5,
                            boxstyle="round,pad=0.05",
                            facecolor=color,
                            edgecolor='darkblue',
                            linewidth=1.5)
        plt.gca().add_patch(box)
        
        # Texte simplifié
        if 'Départ_véhicule_' in node:
            match = re.search(r'Départ_véhicule_(\d+)_(.+)', node)
            if match:
                v_type = match.group(2).split('_')[0][:8]  # Limiter la longueur
                text = f"V{match.group(1)}-{v_type}\n({count})"
            else:
                text = f"{node[:15]}...\n({count})"
        else:
            text = f"{node}\n({count})"
        
        plt.text(x, y, text, 
                ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Dessiner seulement les arêtes principales
    edge_threshold = sorted([G[u][v]['count'] for u, v in G.edges()], reverse=True)
    if len(edge_threshold) > 30:
        edge_threshold = edge_threshold[30]  # Garder seulement les 30 plus fréquentes
    else:
        edge_threshold = 0
    
    for (u, v) in G.edges():
        if u not in pos or v not in pos:
            continue
            
        if G[u][v]['count'] < edge_threshold:
            continue
            
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Données de l'arête
        mean_time = G[u][v]['weight']
        count = G[u][v]['count']
        
        # Largeur proportionnelle mais limitée
        width = min(3, 0.5 + (count / 500))
        
        # Couleur selon le temps
        if mean_time > 30:
            color = 'red'
        elif mean_time > 10:
            color = 'orange'
        else:
            color = 'green'
        
        # Dessiner la flèche
        plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, 
                                  lw=width, alpha=0.5,
                                  connectionstyle="arc3,rad=0.1"))
        
        # Label seulement pour les arêtes importantes
        if count >= n_cases * 0.05:  # Au moins 5% des cas
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            label = f"{mean_time:.0f}m"
            plt.text(mid_x, mid_y, label, 
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.1", 
                            facecolor='white', alpha=0.7),
                    fontsize=6)
    
    # Titre
    plt.title(f"Process Map - {station_name} ({n_cases} interventions)", 
              fontsize=14, fontweight='bold')
    
    # Légende simplifiée
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='green', alpha=0.6, label='< 10 min'),
        Rectangle((0, 0), 1, 1, fc='orange', alpha=0.6, label='10-30 min'),
        Rectangle((0, 0), 1, 1, fc='red', alpha=0.6, label='> 30 min')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Ajuster les marges
    plt.xlim(-2, 16)
    plt.ylim(-5, 5)
    plt.axis('off')
    
    # Sauvegarder avec DPI réduit pour éviter les problèmes de mémoire
    filename = f"process_map_{station_name.replace(' ', '_')}.png"
    
    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.5)
        print(f"✅ Process map sauvegardé: {filename}")
    except MemoryError:
        print("⚠️ Problème de mémoire, sauvegarde avec qualité réduite...")
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.5)
        print(f"✅ Process map sauvegardé (qualité réduite): {filename}")
    finally:
        plt.close()
    
    # Créer un rapport détaillé
    create_enhanced_station_report(station_name, df_station, avg_times, avg_vehicle_times, n_cases)

def create_enhanced_station_report(station_name, df_station, avg_times, avg_vehicle_times, n_cases):
    """
    Créer un rapport détaillé amélioré avec analyse des véhicules
    """
    filename = f"rapport_enhanced_{station_name.replace(' ', '_')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"RAPPORT PROCESS MINING AMÉLIORÉ - CASERNE: {station_name}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Nombre total d'interventions: {n_cases}\n")
        f.write(f"Période: {df_station['timestamp'].min()} à {df_station['timestamp'].max()}\n\n")
        
        # Analyse des véhicules utilisés
        f.write("TYPES DE VÉHICULES UTILISÉS\n")
        f.write("-"*40 + "\n")
        
        vehicle_counts = {}
        for activity in df_station['activity'].unique():
            if 'Départ_véhicule_' in activity:
                match = re.search(r'Départ_véhicule_\d+_(.+)', activity)
                if match:
                    v_type = match.group(1)
                    vehicle_counts[v_type] = vehicle_counts.get(v_type, 0) + 1
        
        for v_type, count in sorted(vehicle_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {v_type}: {count} départs\n")
        
        # Temps par type de véhicule
        f.write("\n\nTEMPS MOYENS PAR TYPE DE VÉHICULE\n")
        f.write("-"*40 + "\n")
        
        for transition, stats in sorted(avg_vehicle_times.items(), 
                                      key=lambda x: x[1]['mean'], reverse=True):
            if stats['count'] >= 5:
                f.write(f"\n{transition}:")
                f.write(f"\n  - Temps moyen: {stats['mean']:.1f} min")
                f.write(f"\n  - Occurrences: {stats['count']}\n")
        
        # Transitions générales
        f.write("\n\nTEMPS MOYENS PAR TRANSITION\n")
        f.write("-"*40 + "\n")
        
        # Filtrer et trier
        sorted_transitions = sorted(avg_times.items(), 
                                  key=lambda x: x[1]['mean'], reverse=True)
        
        for (from_act, to_act), stats in sorted_transitions[:20]:
            if stats['count'] >= 5:
                f.write(f"\n{from_act} → {to_act}:")
                f.write(f"\n  - Temps moyen: {stats['mean']:.1f} min")
                f.write(f"\n  - Temps médian: {stats['median']:.1f} min")
                f.write(f"\n  - Occurrences: {stats['count']}\n")
    
    print(f"✅ Rapport amélioré sauvegardé: {filename}")

def main():
    """
    Fonction principale adaptée pour les logs modifiés
    """
    # Demander quel fichier utiliser
    print("\n📁 Quel fichier de logs voulez-vous utiliser?")
    print("1. event_logs_with_vehicles.csv (logs modifiés avec types)")
    print("2. event_logs_detailles.csv (logs originaux)")
    print("3. Autre fichier")
    
    choice = input("\nChoix (1-3): ")
    
    if choice == '1':
        csv_file = 'event_logs_with_vehicles.csv'
    elif choice == '2':
        csv_file = 'event_logs_detailles.csv'
    else:
        csv_file = input("Nom du fichier CSV: ")
    
    # Charger les données
    df_merged, station_info = load_data_with_stations(csv_file)
    
    # Analyser les activités de véhicules si présentes
    vehicle_activities = analyze_vehicle_activities(df_merged)
    
    # Analyser les casernes
    station_stats = df_merged.groupby('station')['case_id'].nunique().sort_values(ascending=False)
    
    print("\n📊 Casernes disponibles:")
    print(f"Nombre total de casernes: {len(station_stats)}")
    print("\nTop 10 casernes par nombre d'interventions:")
    for station, count in station_stats.head(10).items():
        print(f"  - {station}: {count} interventions")
    
    # Sélection interactive
    while True:
        print("\n🚒 Options:")
        print("1. Analyser une caserne spécifique")
        print("2. Comparer plusieurs casernes")
        print("3. Analyser toutes les données (sans filtre caserne)")
        print("0. Quitter")
        
        option = input("\nChoix: ")
        
        if option == '0':
            break
            
        elif option == '1':
            # Liste des casernes
            stations_list = station_stats.head(20).index.tolist()
            
            print("\nCasernes disponibles:")
            for i, (station, count) in enumerate(zip(stations_list, station_stats.head(20).values)):
                print(f"{i+1:2d}. {station:<30} ({count} interventions)")
            
            choice = input("\n📍 Numéro de la caserne (ou nom complet): ")
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(stations_list):
                    selected_station = stations_list[choice_num - 1]
                    create_enhanced_process_map(selected_station, df_merged)
            except ValueError:
                # C'est un nom
                if choice in station_stats.index:
                    create_enhanced_process_map(choice, df_merged)
                else:
                    print(f"❌ Caserne '{choice}' non trouvée")
                    
        elif option == '2':
            top_stations = station_stats.head(4).index.tolist()
            print(f"\nComparaison des 4 premières casernes: {', '.join(top_stations)}")
            # Implémenter la comparaison si nécessaire
            
        elif option == '3':
            create_enhanced_process_map('TOUTES_CASERNES', df_merged)

if __name__ == "__main__":
    main()