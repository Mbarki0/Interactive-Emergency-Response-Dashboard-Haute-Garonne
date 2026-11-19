import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_real_firefighter_data(pkl_file='df_pc_real.pkl'):
    """
    Analyse directe des données réelles des pompiers
    """
    print("🚒 ANALYSE APPROFONDIE DES INTERVENTIONS POMPIERS")
    print("="*60)
    
    # 1. Charger les données réelles
    print("\n📂 Chargement des données réelles...")
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)
    
    print(f"✅ {len(df)} enregistrements chargés")
    print(f"📊 Colonnes disponibles: {', '.join(df.columns.tolist())}")
    
    # 2. Statistiques générales
    print("\n📈 STATISTIQUES GÉNÉRALES")
    print("-"*50)
    
    # Filtrer les vraies interventions (pas les retours)
    df_interventions = df[df['departure'] != {0: 'RETURN'}].copy()
    print(f"Nombre d'interventions: {len(df_interventions)}")
    print(f"Nombre de retours: {len(df) - len(df_interventions)}")
    
    # 3. Analyse temporelle
    print("\n⏰ ANALYSE TEMPORELLE")
    print("-"*50)
    
    # Distribution par mois
    month_counts = df_interventions['Month'].value_counts().sort_index()
    print("\nInterventions par mois:")
    for month, count in month_counts.items():
        print(f"  Mois {month}: {count} interventions")
    
    # Distribution par heure
    hour_counts = df_interventions['Hour'].value_counts().sort_index()
    peak_hour = hour_counts.idxmax()
    print(f"\nHeure de pointe: {peak_hour}h avec {hour_counts[peak_hour]} interventions")
    
    # 4. Analyse spatiale
    print("\n📍 ANALYSE SPATIALE")
    print("-"*50)
    
    if 'zone' in df_interventions.columns:
        zone_counts = df_interventions['zone'].value_counts()
        print("\nTop 5 des zones d'intervention:")
        for zone, count in zone_counts.head().items():
            print(f"  - {zone}: {count} interventions ({count/len(df_interventions)*100:.1f}%)")
    
    # 5. Analyse des types d'incidents
    print("\n🚨 TYPES D'INCIDENTS")
    print("-"*50)
    
    if 'Incident' in df_interventions.columns:
        incident_counts = df_interventions['Incident'].value_counts()
        print(f"\nNombre de types d'incidents: {len(incident_counts)}")
        print("\nTop 10 des types d'incidents:")
        for incident, count in incident_counts.head(10).items():
            print(f"  - Type {incident}: {count} fois ({count/len(df_interventions)*100:.1f}%)")
    
    # 6. Analyse des durées
    print("\n⏱️ ANALYSE DES DURÉES")
    print("-"*50)
    
    durations = df_interventions['Duration']
    print(f"Durée moyenne: {durations.mean():.1f} minutes")
    print(f"Durée médiane: {durations.median():.1f} minutes")
    print(f"Durée min/max: {durations.min():.1f} / {durations.max():.1f} minutes")
    print(f"Écart-type: {durations.std():.1f} minutes")
    
    # Durées par type d'incident
    if 'Incident' in df_interventions.columns:
        print("\nDurée moyenne par type d'incident (Top 5):")
        incident_durations = df_interventions.groupby('Incident')['Duration'].agg(['mean', 'count'])
        incident_durations = incident_durations[incident_durations['count'] > 100].sort_values('mean', ascending=False)
        for incident, row in incident_durations.head().iterrows():
            print(f"  - Type {incident}: {row['mean']:.1f} min (n={row['count']})")
    
    # 7. Analyse des départs
    print("\n🚗 ANALYSE DES DÉPARTS")
    print("-"*50)
    
    # Compter les véhicules par intervention
    vehicle_counts = []
    for dep in df_interventions['departure']:
        if isinstance(dep, dict):
            vehicle_counts.append(len(dep))
    
    vehicle_series = pd.Series(vehicle_counts)
    print(f"\nNombre moyen de véhicules par intervention: {vehicle_series.mean():.2f}")
    print("Distribution des départs:")
    for n_vehicles, count in vehicle_series.value_counts().sort_index().items():
        print(f"  - {n_vehicles} véhicule(s): {count} interventions ({count/len(vehicle_series)*100:.1f}%)")
    
    # 8. Process Mining - Créer des traces réalistes
    print("\n🔄 CRÉATION DES TRACES DE PROCESSUS")
    print("-"*50)
    
    event_logs = []
    
    for idx, row in df_interventions.iterrows():
        num_inter = row['num_inter']
        start_time = row['date']
        duration = row['Duration']
        departure = row['departure']
        
        # Créer une séquence d'événements réaliste
        events = []
        
        # 1. Appel reçu
        events.append({
            'case_id': str(num_inter),
            'activity': 'Appel_reçu',
            'timestamp': start_time,
            'incident_type': row.get('Incident', ''),
            'zone': row.get('zone', '')
        })
        
        # 2. Analyse et décision
        if isinstance(departure, dict) and len(departure) > 0:
            # Temps de décision variable selon le type d'incident
            decision_delay = 2 if row.get('Incident', 0) < 10 else 3
            events.append({
                'case_id': str(num_inter),
                'activity': 'Décision_départ',
                'timestamp': start_time + pd.Timedelta(minutes=decision_delay),
                'vehicles_count': len(departure)
            })
            
            # 3. Départ des véhicules
            for i, (num, vehicles) in enumerate(departure.items()):
                events.append({
                    'case_id': str(num_inter),
                    'activity': f'Départ_véhicule_{i+1}',
                    'timestamp': start_time + pd.Timedelta(minutes=5+i),
                    'vehicle_type': str(vehicles)
                })
        
        # 4. Arrivée sur site
        arrival_time = 15 if row.get('zone', '') == '' else 20
        events.append({
            'case_id': str(num_inter),
            'activity': 'Arrivée_sur_site',
            'timestamp': start_time + pd.Timedelta(minutes=arrival_time)
        })
        
        # 5. Intervention
        events.append({
            'case_id': str(num_inter),
            'activity': 'Intervention_en_cours',
            'timestamp': start_time + pd.Timedelta(minutes=arrival_time+5)
        })
        
        # 6. Fin d'intervention
        events.append({
            'case_id': str(num_inter),
            'activity': 'Fin_intervention',
            'timestamp': start_time + pd.Timedelta(minutes=duration)
        })
        
        event_logs.extend(events)
    
    # Convertir en DataFrame
    df_events = pd.DataFrame(event_logs)
    
    # Sauvegarder pour PM4Py
    df_events.to_csv('event_logs_detailles.csv', index=False)
    print(f"✅ {len(df_events)} événements créés et sauvegardés dans 'event_logs_detailles.csv'")
    
    # 9. Visualisations
    print("\n📊 CRÉATION DES VISUALISATIONS")
    print("-"*50)
    
    # Graphique 1: Interventions par heure
    plt.figure(figsize=(12, 6))
    hour_counts.plot(kind='bar', color='steelblue')
    plt.title('Distribution des interventions par heure de la journée')
    plt.xlabel('Heure')
    plt.ylabel('Nombre d\'interventions')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('interventions_par_heure.png')
    plt.close()
    print("✅ Graphique sauvegardé: interventions_par_heure.png")
    
    # Graphique 2: Durées par type d'incident
    if 'Incident' in df_interventions.columns:
        plt.figure(figsize=(10, 8))
        top_incidents = incident_counts.head(10).index
        df_top = df_interventions[df_interventions['Incident'].isin(top_incidents)]
        df_top.boxplot(column='Duration', by='Incident', figsize=(12, 8))
        plt.title('Distribution des durées par type d\'incident (Top 10)')
        plt.xlabel('Type d\'incident')
        plt.ylabel('Durée (minutes)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('durees_par_incident.png')
        plt.close()
        print("✅ Graphique sauvegardé: durees_par_incident.png")
    
    # 10. Rapport final
    with open('analyse_interventions_pompiers.txt', 'w', encoding='utf-8') as f:
        f.write("ANALYSE APPROFONDIE DES INTERVENTIONS POMPIERS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Période analysée: {df_interventions['date'].min()} à {df_interventions['date'].max()}\n")
        f.write(f"Nombre total d'interventions: {len(df_interventions)}\n")
        f.write(f"Durée moyenne: {durations.mean():.1f} minutes\n")
        f.write(f"Pic d'activité: {peak_hour}h\n")
        
        if 'zone' in df_interventions.columns:
            f.write(f"\nZone la plus active: {zone_counts.index[0]} ({zone_counts.iloc[0]} interventions)\n")
        
        f.write("\n\nRÉPARTITION MENSUELLE:\n")
        for month, count in month_counts.items():
            f.write(f"Mois {month}: {count} interventions\n")
    
    print("✅ Rapport sauvegardé: analyse_interventions_pompiers.txt")
    
    print("\n✨ ANALYSE TERMINÉE!")
    print("\nFichiers créés:")
    print("  - event_logs_detailles.csv : Logs détaillés pour PM4Py")
    print("  - interventions_par_heure.png : Distribution horaire")
    print("  - durees_par_incident.png : Durées par type")
    print("  - analyse_interventions_pompiers.txt : Rapport complet")

if __name__ == "__main__":
    analyze_real_firefighter_data('df_pc_real.pkl')