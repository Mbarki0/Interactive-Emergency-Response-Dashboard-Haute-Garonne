import pandas as pd
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def complete_process_mining_analysis(csv_file='event_logs_detailles.csv'):
    """
    Analyse complète de Process Mining pour les interventions pompiers
    """
    print("🚒 ANALYSE PROCESS MINING COMPLÈTE - INTERVENTIONS POMPIERS")
    print("="*70)
    
    # 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
    print("\n📂 1. CHARGEMENT DES DONNÉES")
    print("-"*50)
    
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Renommer pour PM4Py
    df = df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    
    # Garder les attributs supplémentaires s'ils existent
    attribute_columns = []
    for col in ['incident_type', 'zone', 'vehicles_count', 'vehicle_type']:
        if col in df.columns:
            attribute_columns.append(col)
    
    print(f"✅ {len(df)} événements chargés")
    print(f"✅ {df['case:concept:name'].nunique()} cas uniques")
    print(f"✅ {df['concept:name'].nunique()} activités différentes")
    
    # Convertir en event log
    event_log = pm4py.format_dataframe(df, 
                                      case_id='case:concept:name',
                                      activity_key='concept:name',
                                      timestamp_key='time:timestamp')
    
    # 2. STATISTIQUES DE BASE
    print("\n📊 2. STATISTIQUES DE BASE")
    print("-"*50)
    
    # Activités
    activities = pm4py.get_event_attribute_values(event_log, 'concept:name')
    print("\nActivités dans le processus:")
    for activity, count in sorted(activities.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {activity}: {count} occurrences")
    
    # Durées moyennes
    print("\n⏱️ Durées des cas:")
    all_case_durations = case_statistics.get_all_case_durations(event_log, 
                                                                parameters={
                                                                    case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"
                                                                })
    durations_minutes = [d/60 for d in all_case_durations]
    print(f"  - Moyenne: {sum(durations_minutes)/len(durations_minutes):.1f} minutes")
    print(f"  - Min: {min(durations_minutes):.1f} minutes")
    print(f"  - Max: {max(durations_minutes):.1f} minutes")
    
    # 3. ANALYSE DES VARIANTES
    print("\n🔄 3. ANALYSE DES VARIANTES DE PROCESSUS")
    print("-"*50)
    
    variants = pm4py.get_variants(event_log)
    print(f"\nNombre de variantes uniques: {len(variants)}")
    
    # Analyser les variantes
    variants_count = {}
    for variant in variants:
        if isinstance(variants[variant], list):
            variants_count[variant] = len(variants[variant])
        else:
            variants_count[variant] = variants[variant]
    
    # Top variantes
    sorted_variants = sorted(variants_count.items(), key=lambda x: x[1], reverse=True)
    total_cases = sum(count for _, count in sorted_variants)
    
    print("\n📌 Top 10 des variantes les plus fréquentes:")
    for i, (variant, count) in enumerate(sorted_variants[:10]):
        percentage = (count / total_cases) * 100
        # Gérer différents formats de variantes
        if isinstance(variant, tuple):
            activities = list(variant)
        elif isinstance(variant, str):
            activities = variant.split(',')
        else:
            activities = [str(variant)]
        print(f"\n{i+1}. Variante ({count} cas - {percentage:.2f}%):")
        print(f"   {' → '.join(activities)}")
    
    # 4. DÉCOUVERTE DE PROCESSUS - MULTIPLE ALGORITHMES
    print("\n🔍 4. DÉCOUVERTE DE MODÈLES DE PROCESSUS")
    print("-"*50)
    
    # 4.1 Directly-Follows Graph (DFG)
    print("\n📈 4.1 Directly-Follows Graph")
    dfg = dfg_discovery.apply(event_log)
    start_activities = pm4py.get_start_activities(event_log)
    end_activities = pm4py.get_end_activities(event_log)
    
    # Visualiser et sauvegarder DFG
    gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.FREQUENCY,
                                   parameters={"start_activities": start_activities,
                                             "end_activities": end_activities})
    dfg_visualization.save(gviz, "process_mining_dfg.png")
    print("✅ DFG sauvegardé: process_mining_dfg.png")
    
    # 4.2 Heuristics Miner
    print("\n🔧 4.2 Heuristics Miner")
    try:
        heu_net = heuristics_miner.apply(event_log, 
                                         parameters={
                                             heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
                                             heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.65,
                                             heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: 30,
                                             heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 30
                                         })
        gviz = hn_visualizer.apply(heu_net)
        hn_visualizer.save(gviz, "process_mining_heuristics.png")
        print("✅ Modèle heuristique sauvegardé: process_mining_heuristics.png")
    except Exception as e:
        print(f"⚠️ Erreur Heuristics Miner: {str(e)}")
    
    # 4.3 Inductive Miner
    print("\n🌳 4.3 Inductive Miner")
    try:
        tree = inductive_miner.apply(event_log)
        gviz = pt_visualizer.apply(tree)
        pt_visualizer.save(gviz, "process_mining_tree.png")
        print("✅ Process tree sauvegardé: process_mining_tree.png")
        
        # Convertir en Petri Net pour analyse
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
        gviz = pn_visualizer.apply(net, initial_marking, final_marking)
        pn_visualizer.save(gviz, "process_mining_petri.png")
        print("✅ Petri net sauvegardé: process_mining_petri.png")
    except Exception as e:
        print(f"⚠️ Erreur Inductive Miner: {str(e)}")
    
    # 5. ANALYSE DE PERFORMANCE
    print("\n⚡ 5. ANALYSE DE PERFORMANCE")
    print("-"*50)
    
    # Performance par activité
    print("\n📊 Performance par activité:")
    activity_performance = {}
    
    # Analyser directement depuis le dataframe pour éviter les problèmes de format
    for case_id, group in df.groupby('case:concept:name'):
        group = group.sort_values('time:timestamp')
        for i in range(len(group)-1):
            current_act = group.iloc[i]['concept:name']
            next_act = group.iloc[i+1]['concept:name']
            time_diff = (group.iloc[i+1]['time:timestamp'] - group.iloc[i]['time:timestamp']).total_seconds() / 60
            
            key = f"{current_act} → {next_act}"
            if key not in activity_performance:
                activity_performance[key] = []
            activity_performance[key].append(time_diff)
    
    # Calculer moyennes et trier
    avg_performance = {}
    for transition, times in activity_performance.items():
        if len(times) > 10:  # Au moins 10 occurrences
            avg_performance[transition] = {
                'avg': sum(times) / len(times),
                'count': len(times)
            }
    
    # Top 10 transitions les plus lentes
    sorted_perf = sorted(avg_performance.items(), key=lambda x: x[1]['avg'], reverse=True)
    print("\n🐌 Top 10 des transitions les plus lentes:")
    for transition, stats in sorted_perf[:10]:
        print(f"  - {transition}: {stats['avg']:.1f} min en moyenne ({stats['count']} fois)")
    
    # 6. ANALYSE DES CHEMINS CRITIQUES
    print("\n🛤️ 6. ANALYSE DES CHEMINS CRITIQUES")
    print("-"*50)
    
    # Identifier les chemins les plus longs depuis le dataframe
    longest_cases = []
    for case_id, group in df.groupby('case:concept:name'):
        group = group.sort_values('time:timestamp')
        if len(group) > 0:
            duration = (group.iloc[-1]['time:timestamp'] - group.iloc[0]['time:timestamp']).total_seconds() / 60
            activities = group['concept:name'].tolist()
            longest_cases.append({
                'case': case_id,
                'duration': duration,
                'length': len(group),
                'path': ' → '.join(activities)
            })
    
    # Trier par durée
    longest_cases.sort(key=lambda x: x['duration'], reverse=True)
    
    print("\n⏰ Top 5 des cas les plus longs:")
    for i, case in enumerate(longest_cases[:5]):
        print(f"\n{i+1}. Cas {case['case']}")
        print(f"   Durée: {case['duration']:.1f} minutes")
        print(f"   Nombre d'étapes: {case['length']}")
        print(f"   Chemin: {case['path'][:100]}...")
    
    # 7. ANALYSE PAR ATTRIBUTS
    print("\n🏷️ 7. ANALYSE PAR ATTRIBUTS")
    print("-"*50)
    
    # Si on a des attributs supplémentaires
    if 'incident_type' in df.columns:
        print("\n🚨 Analyse par type d'incident:")
        incident_variants = {}
        
        # Analyser directement depuis le dataframe
        for case_id, group in df.groupby('case:concept:name'):
            # Récupérer le type d'incident du premier événement
            incident_type = None
            if 'incident_type' in group.columns and len(group) > 0:
                incident_type = group.iloc[0]['incident_type']
            
            if incident_type and pd.notna(incident_type):
                # Créer la variante comme string
                activities = group.sort_values('time:timestamp')['concept:name'].tolist()
                variant = ' → '.join(activities)
                
                if incident_type not in incident_variants:
                    incident_variants[incident_type] = {}
                if variant not in incident_variants[incident_type]:
                    incident_variants[incident_type][variant] = 0
                incident_variants[incident_type][variant] += 1
        
        # Afficher les variantes dominantes par type
        for incident_type, variants in list(incident_variants.items())[:5]:
            print(f"\n  Type {incident_type}:")
            sorted_vars = sorted(variants.items(), key=lambda x: x[1], reverse=True)
            dominant = sorted_vars[0] if sorted_vars else ('', 0)
            print(f"    Variante dominante: {dominant[0][:50]}... ({dominant[1]} cas)")
    
    # 8. CONFORMANCE CHECKING
    print("\n✅ 8. ANALYSE DE CONFORMITÉ")
    print("-"*50)
    
    try:
        # Utiliser l'Inductive Miner pour avoir un modèle
        print("Création du modèle de référence...")
        tree = inductive_miner.apply(event_log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)
        
        # Token replay pour la conformité
        aligned_traces = token_replay.apply(event_log, net, initial_marking, final_marking)
        
        if aligned_traces:
            fitness_values = []
            for trace in aligned_traces:
                if 'trace_fitness' in trace:
                    fitness_values.append(trace['trace_fitness'])
            
            if fitness_values:
                avg_fitness = sum(fitness_values) / len(fitness_values)
                print(f"\n📏 Fitness moyen du modèle: {avg_fitness:.2%}")
                print(f"   - Traces parfaitement conformes: {sum(1 for f in fitness_values if f == 1.0)}")
                print(f"   - Traces non conformes: {sum(1 for f in fitness_values if f < 1.0)}")
            else:
                print("⚠️ Pas de valeurs de fitness disponibles")
    except Exception as e:
        print(f"⚠️ Conformance checking non disponible: {str(e)}")
    
    # 9. EXPORT ET RAPPORT FINAL
    print("\n📝 9. GÉNÉRATION DU RAPPORT FINAL")
    print("-"*50)
    
    with open('rapport_process_mining_complet.txt', 'w', encoding='utf-8') as f:
        f.write("RAPPORT COMPLET PROCESS MINING - INTERVENTIONS POMPIERS\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. RÉSUMÉ EXÉCUTIF\n")
        f.write("-"*30 + "\n")
        f.write(f"Nombre total d'interventions analysées: {df['case:concept:name'].nunique()}\n")
        f.write(f"Nombre total d'événements: {len(df)}\n")
        f.write(f"Nombre d'activités uniques: {df['concept:name'].nunique()}\n")
        f.write(f"Nombre de variantes de processus: {len(variants)}\n")
        f.write(f"Durée moyenne des interventions: {sum(durations_minutes)/len(durations_minutes):.1f} minutes\n")
        
        f.write("\n2. VARIANTES PRINCIPALES\n")
        f.write("-"*30 + "\n")
        for i, (variant, count) in enumerate(sorted_variants[:5]):
            percentage = (count / total_cases) * 100
            if isinstance(variant, tuple):
                variant_str = ' → '.join(variant)
            elif isinstance(variant, str):
                variant_str = variant
            else:
                variant_str = str(variant)
            f.write(f"{i+1}. {variant_str} ({count} cas - {percentage:.2f}%)\n")
        
        f.write("\n3. GOULOTS D'ÉTRANGLEMENT\n")
        f.write("-"*30 + "\n")
        for transition, stats in sorted_perf[:5]:
            f.write(f"- {transition}: {stats['avg']:.1f} min en moyenne\n")
        
        f.write("\n4. RECOMMANDATIONS\n")
        f.write("-"*30 + "\n")
        f.write("- Optimiser les transitions les plus lentes identifiées\n")
        f.write("- Standardiser les variantes de processus pour réduire la complexité\n")
        f.write("- Investiguer les cas avec durées extrêmes\n")
    
    print("✅ Rapport complet sauvegardé: rapport_process_mining_complet.txt")
    
    # 10. VISUALISATIONS SUPPLÉMENTAIRES
    print("\n📊 10. CRÉATION DE VISUALISATIONS SUPPLÉMENTAIRES")
    print("-"*50)
    
    # Graphique des variantes
    plt.figure(figsize=(12, 6))
    variant_names = [f"Var{i+1}" for i in range(min(20, len(sorted_variants)))]
    variant_counts = [count for _, count in sorted_variants[:20]]
    plt.bar(variant_names, variant_counts, color='steelblue')
    plt.xlabel('Variantes de processus')
    plt.ylabel('Nombre de cas')
    plt.title('Distribution des 20 principales variantes de processus')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('variantes_distribution.png')
    plt.close()
    print("✅ Graphique sauvegardé: variantes_distribution.png")
    
    # Heatmap des transitions
    if len(activity_performance) > 0:
        # Créer une matrice pour la heatmap
        # Récupérer la liste des activités depuis le début de l'analyse
        activities_unique = df['concept:name'].unique().tolist()
        matrix = pd.DataFrame(0.0, index=activities_unique, columns=activities_unique)
        
        for transition, times in activity_performance.items():
            if ' → ' in transition:
                from_act, to_act = transition.split(' → ')
                if from_act in activities_unique and to_act in activities_unique:
                    matrix.loc[from_act, to_act] = sum(times) / len(times) if times else 0
        
        # Créer la heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(matrix, cmap='YlOrRd', cbar_kws={'label': 'Temps moyen (minutes)'})
        plt.title('Heatmap des temps de transition entre activités')
        plt.xlabel('Activité suivante')
        plt.ylabel('Activité précédente')
        plt.tight_layout()
        plt.savefig('heatmap_transitions.png')
        plt.close()
        print("✅ Heatmap sauvegardée: heatmap_transitions.png")
    
    print("\n✨ ANALYSE PROCESS MINING COMPLÈTE TERMINÉE!")
    print("\n📁 Fichiers générés:")
    print("  - process_mining_dfg.png : Graphe des flux directs")
    print("  - process_mining_heuristics.png : Modèle heuristique")
    print("  - process_mining_tree.png : Arbre de processus")
    print("  - process_mining_petri.png : Réseau de Petri")
    print("  - variantes_distribution.png : Distribution des variantes")
    print("  - heatmap_transitions.png : Heatmap des transitions")
    print("  - rapport_process_mining_complet.txt : Rapport détaillé")

if __name__ == "__main__":
    # Utiliser le fichier généré par l'analyse précédente
    complete_process_mining_analysis('event_logs_detailles.csv')