import gradio as gr
import folium
import os 
import plotly.graph_objects as go

def run_simulation(duration):
    return f"Simulation run for {duration}"

def filter_data(date, incident_type):
    return f"Filtered data for date: {date} and incident type: {incident_type}"

def generate_haute_garonne_map_plotly():
    # Rectangle approximatif autour de la Haute-Garonne
    lats = [43.1, 43.1, 44.0, 44.0, 43.1]
    lons = [0.9, 2.1, 2.1, 0.9, 0.9]
    fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='lines',
        fill='toself',
        marker={'size': 10},
        name="Haute-Garonne"
    ))
    # Marqueur Toulouse
    fig.add_trace(go.Scattermapbox(
        lat=[43.6045],
        lon=[1.4442],
        mode='markers+text',
        marker={'size': 14, 'color': 'red'},
        text=["Toulouse"],
        textposition="top right",
        name="Toulouse"
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=8.5,
        mapbox_center={"lat": 43.6045, "lon": 1.4442},
        margin={"r":0,"t":0,"l":0,"b":0},
        height=500
    )
    return fig

def display_map():
    # Example function to display a map
    return "Map content goes here"

def display_overview():
    # Example function to display an overview
    return "Overview content goes here"

def display_curve_1():
    return "Curve 1 content goes here"

def display_curve_2():
    return "Curve 2 content goes here"

def display_curve_3():
    return "Curve 3 content goes here"

with gr.Blocks() as app:
    gr.Markdown("## Dashboard de Simulation et Visualisation")

    with gr.Row():
        year_input = gr.Textbox(label="Enter Year", value="0")
        month_input = gr.Textbox(label="Enter Month", value="0")
        day_input = gr.Textbox(label="Enter Day", value="0")
        run_button = gr.Button("Run")
        simulation_output = gr.Markdown()
        run_button.click(fn=run_simulation, inputs=[year_input, month_input, day_input], outputs=simulation_output)

    with gr.Tabs():
        with gr.Tab("Map"):
            gr.Markdown("### 🗺️ Interactive Map")
            gr.Markdown("#### Map Display")
            map_plot = gr.Plot()
            show_map_btn = gr.Button("Afficher la carte Haute-Garonne")
            show_map_btn.click(fn=generate_haute_garonne_map_plotly, outputs=map_plot)

        with gr.Tab("Overview"):
            gr.Markdown("### 📊 Aperçu des données")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Filter Options")

                    # Filter 1: Date selection
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("##### Select Filter")
                            year_input = gr.Dropdown(label="Filter", choices=["hour", "day", "month", "year"], value="hour")
                        with gr.Column():
                            gr.Markdown("##### Select Incident Type")
                            incident_type_input = gr.Dropdown(
                                label="Incident Type",
                                choices=["Type A", "Type B", "Type C", "All"],
                                value="Type A"
                            )

                    # Filter button
                    filter_button = gr.Button("Filter Data")
                    filter_output = gr.Markdown()
                    filter_button.click(
                        fn=filter_data,
                        inputs=[year_input, month_input, day_input, incident_type_input],
                        outputs=filter_output
                    )

    
                    gr.Markdown("#### Curve Display")
                    
                    # Trois grands rectangles vides
                    gr.Image(value="default1.png", label="Courbe 1", elem_id="curve-img-1", height=500)
                    gr.Image(value="default2.png", label="Courbe 2", elem_id="curve-img-2", height=500)
                    gr.Image(value="default3.png", label="Courbe 3", elem_id="curve-img-3", height=500)

        with gr.Tab("Logs"):
                gr.Markdown("### 🗺️ Logs de la simulation")
                map_output = gr.Markdown()


app.launch()