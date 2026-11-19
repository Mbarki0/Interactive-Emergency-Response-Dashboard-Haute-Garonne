import pandas as pd
import pickle
import re

def extract_vehicle_types_from_departure(departure_dict):
    """
    Extraire les types de véhicules depuis le dictionnaire departure
    """
    vehicle_types = []
    
    if isinstance(departure_dict, dict):
        for num, vehicles in sorted(departure_dict.items()):
            if isinstance(vehicles, list):
                # Joindre les types de véhicules
                vehicle_type = '_'.join(vehicles)
                vehicle_types.append(vehicle_type)
            else:
                vehicle_types.append(str(vehicles))
    
    return vehicle_types

def modify_event_logs_with_vehicle_types(csv_input='event_logs_detailles.csv', 
                                        pkl_file='df_pc_real.pkl',
                                        csv_output='event_logs_with_vehicles.csv'):
    """
    Modifier les logs pour inclure les vrais types de véhicules
    """
    print("🚒 MODIFICATION DES LOGS AVEC TYPES DE VÉHICULES")
    print("="*60)
    
    # 1. Charger les données
    print("\n📂 Chargement des données...")
    
    # Charger les logs
    df_logs = pd.read_csv(csv_input)
    print(f"✅ {len(df_logs)} événements chargés")
    
    # Charger les données réelles pour avoir les types de véhicules
    with open(pkl_file, 'rb') as f:
        df_real = pickle.load(f)
    print(f"✅ {len(df_real)} interventions avec détails chargées")
    
    # 2. Créer un mapping intervention -> types de véhicules
    print("\n🔄 Création du mapping des véhicules...")
    
    vehicle_mapping = {}
    for idx, row in df_real.iterrows():
        num_inter = row['num_inter']
        departure = row['departure']
        
        if isinstance(departure, dict) and departure != {0: 'RETURN'}:
            vehicle_types = extract_vehicle_types_from_departure(departure)
            vehicle_mapping[num_inter] = vehicle_types
    
    print(f"✅ Mapping créé pour {len(vehicle_mapping)} interventions")
    
    # 3. Afficher quelques exemples de types de véhicules trouvés
    print("\n📋 Exemples de types de véhicules trouvés:")
    vehicle_examples = set()
    for vehicles in list(vehicle_mapping.values())[:100]:
        vehicle_examples.update(vehicles)
    
    for v_type in sorted(list(vehicle_examples))[:20]:
        print(f"  - {v_type}")
    
    # 4. Modifier les logs
    print("\n✏️ Modification des logs...")
    print("Format: Départ_véhicule_[numéro]_[type]")
    print("Exemple: Départ_véhicule_1_FPT")
    
    df_logs_modified = df_logs.copy()
    modifications = 0
    
    for idx, row in df_logs_modified.iterrows():
        case_id = int(row['case_id'])
        activity = row['activity']
        
        # Si c'est une activité de départ de véhicule
        if 'Départ_véhicule_' in activity:
            # Extraire le numéro du véhicule
            vehicle_num_match = re.search(r'Départ_véhicule_(\d+)', activity)
            
            if vehicle_num_match and case_id in vehicle_mapping:
                vehicle_index = int(vehicle_num_match.group(1)) - 1
                vehicle_types = vehicle_mapping[case_id]
                
                if 0 <= vehicle_index < len(vehicle_types):
                    # Remplacer par le type réel AVEC le numéro
                    vehicle_num = vehicle_num_match.group(1)
                    new_activity = f"Départ_véhicule_{vehicle_num}_{vehicle_types[vehicle_index]}"
                    df_logs_modified.at[idx, 'activity'] = new_activity
                    
                    # Ajouter le type de véhicule comme colonne supplémentaire
                    df_logs_modified.at[idx, 'vehicle_type'] = vehicle_types[vehicle_index]
                    df_logs_modified.at[idx, 'vehicle_number'] = int(vehicle_num)
                    modifications += 1
    
    print(f"✅ {modifications} activités modifiées")
    
    # 5. Statistiques sur les modifications
    print("\n📊 Statistiques des modifications:")
    
    # Compter les nouveaux types d'activités
    new_activities = df_logs_modified[df_logs_modified['activity'].str.startswith('Départ_')]['activity'].value_counts()
    
    print("\nTop 20 des départs par type de véhicule:")
    for activity, count in new_activities.head(20).items():
        print(f"  - {activity}: {count} occurrences")
    
    # 6. Sauvegarder le fichier modifié
    df_logs_modified.to_csv(csv_output, index=False)
    print(f"\n✅ Fichier modifié sauvegardé: {csv_output}")
    
    # 7. Créer un rapport de synthèse
    create_modification_report(df_logs, df_logs_modified, modifications)
    
    return df_logs_modified

def create_modification_report(df_original, df_modified, n_modifications):
    """
    Créer un rapport sur les modifications effectuées
    """
    with open('rapport_modifications_vehicules.txt', 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE MODIFICATION DES LOGS - TYPES DE VÉHICULES\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Nombre total d'événements: {len(df_original)}\n")
        f.write(f"Nombre de modifications: {n_modifications}\n")
        f.write(f"Pourcentage modifié: {n_modifications/len(df_original)*100:.1f}%\n\n")
        
        # Activités originales
        f.write("ACTIVITÉS ORIGINALES:\n")
        f.write("-"*30 + "\n")
        orig_activities = df_original['activity'].value_counts()
        for act, count in orig_activities.items():
            f.write(f"{act}: {count}\n")
        
        # Nouvelles activités
        f.write("\n\nNOUVELLES ACTIVITÉS (Départs uniquement):\n")
        f.write("-"*30 + "\n")
        new_departs = df_modified[df_modified['activity'].str.startswith('Départ_')]['activity'].value_counts()
        for act, count in new_departs.items():
            f.write(f"{act}: {count}\n")
    
    print("✅ Rapport sauvegardé: rapport_modifications_vehicules.txt")

def analyze_vehicle_patterns(df_modified):
    """
    Analyser les patterns de véhicules après modification
    """
    print("\n🔍 ANALYSE DES PATTERNS DE VÉHICULES")
    print("="*50)
    
    # Types de véhicules uniques
    vehicle_activities = df_modified[df_modified['activity'].str.startswith('Départ_véhicule_')]
    unique_vehicles = set()
    vehicle_order_stats = {}
    
    for activity in vehicle_activities['activity'].unique():
        # Extraire le numéro et le type
        match = re.search(r'Départ_véhicule_(\d+)_(.+)', activity)
        if match:
            vehicle_num = int(match.group(1))
            vehicle_type = match.group(2)
            unique_vehicles.add(vehicle_type)
            
            # Statistiques par position
            if vehicle_num not in vehicle_order_stats:
                vehicle_order_stats[vehicle_num] = {}
            if vehicle_type not in vehicle_order_stats[vehicle_num]:
                vehicle_order_stats[vehicle_num][vehicle_type] = 0
            vehicle_order_stats[vehicle_num][vehicle_type] += 1
    
    print(f"\nNombre de types de véhicules uniques: {len(unique_vehicles)}")
    
    # Afficher les statistiques par position
    print("\n📊 Types de véhicules par position:")
    for vehicle_num in sorted(vehicle_order_stats.keys()):
        print(f"\n🚒 Véhicule {vehicle_num}:")
        sorted_types = sorted(vehicle_order_stats[vehicle_num].items(), 
                            key=lambda x: x[1], reverse=True)
        for v_type, count in sorted_types[:5]:
            print(f"  - {v_type}: {count} fois")
    
    # Combinaisons fréquentes
    print("\n🚗 Combinaisons de véhicules fréquentes:")
    
    vehicle_combinations = {}
    for case_id, group in df_modified.groupby('case_id'):
        vehicles = []
        for activity in group['activity']:
            if activity.startswith('Départ_véhicule_'):
                match = re.search(r'Départ_véhicule_\d+_(.+)', activity)
                if match:
                    vehicle_type = match.group(1)
                    vehicles.append(vehicle_type)
        
        if vehicles:
            combo = ' + '.join(sorted(vehicles))
            vehicle_combinations[combo] = vehicle_combinations.get(combo, 0) + 1
    
    # Top 15 combinaisons
    sorted_combos = sorted(vehicle_combinations.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sorted_combos[:15]:
        print(f"  - {combo}: {count} fois")
    
    return unique_vehicles, vehicle_combinations

def create_enhanced_process_map(df_modified, station=None, save_name='process_map_enhanced.png'):
    """
    Créer un process map amélioré avec les vrais types de véhicules
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    print(f"\n🗺️ Création du Process Map amélioré...")
    
    # Filtrer par station si spécifié
    if station:
        # Il faudrait avoir la colonne station dans df_modified
        print(f"Filtrage pour la station: {station}")
    
    # Calculer les transitions
    transitions = {}
    for case_id, group in df_modified.groupby('case_id'):
        group = group.sort_values('timestamp')
        for i in range(len(group) - 1):
            current = group.iloc[i]['activity']
            next_act = group.iloc[i + 1]['activity']
            key = (current, next_act)
            transitions[key] = transitions.get(key, 0) + 1
    
    # Créer le graphe
    G = nx.DiGraph()
    
    # Ajouter les nœuds et arêtes
    for (from_act, to_act), count in transitions.items():
        if count > 10:  # Seuil pour la lisibilité
            G.add_edge(from_act, to_act, weight=count)
    
    # Visualisation
    plt.figure(figsize=(20, 14))
    
    # Layout personnalisé
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Ajuster les positions pour les activités principales
    fixed_positions = {
        'Appel_reçu': (-3, 0),
        'Décision_départ': (-1, 0),
        'Arrivée_sur_site': (1, 0),
        'Intervention_en_cours': (2, 0),
        'Fin_intervention': (3, 0)
    }
    
    for node, position in fixed_positions.items():
        if node in pos:
            pos[node] = position
    
    # Dessiner
    node_sizes = [G.degree(node) * 100 for node in G.nodes()]
    
    # Couleurs par type d'activité
    node_colors = []
    for node in G.nodes():
        if node.startswith('Départ_'):
            node_colors.append('lightcoral')
        elif node in ['Appel_reçu', 'Décision_départ']:
            node_colors.append('lightgreen')
        elif node in ['Fin_intervention']:
            node_colors.append('lightblue')
        else:
            node_colors.append('lightyellow')
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Arêtes avec épaisseur proportionnelle
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    edge_widths = [w/max_weight * 5 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, arrowsize=20)
    
    plt.title("Process Map avec Types de Véhicules Réels", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Process map amélioré sauvegardé: {save_name}")

def main():
    """
    Fonction principale
    """
    # 1. Modifier les logs
    df_modified = modify_event_logs_with_vehicle_types()
    
    # 2. Analyser les patterns
    unique_vehicles, combinations = analyze_vehicle_patterns(df_modified)
    
    # 3. Créer un process map amélioré
    create_enhanced_process_map(df_modified)
    
    # 4. Option pour analyser par station
    analyze_by_station = input("\n📍 Voulez-vous analyser une station spécifique? (o/n): ")
    if analyze_by_station.lower() == 'o':
        # Il faudrait d'abord ajouter les informations de station aux logs modifiés
        print("Cette fonctionnalité nécessite d'ajouter les informations de station aux logs.")
    
    print("\n✨ Modification terminée!")
    print(f"\nUtilisez maintenant '{csv_output}' pour vos analyses Process Mining")
    print("Les activités 'Départ_véhicule_X' ont été remplacées par les vrais types!")

if __name__ == "__main__":
    # Variables globales pour les noms de fichiers
    csv_output = 'event_logs_with_vehicles.csv'
    main()