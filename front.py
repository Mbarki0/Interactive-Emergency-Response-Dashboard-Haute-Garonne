import gradio as gr

def run_simulation(duration):
    return f"Simulation run for {duration}"

def filter_data(date, incident_type):
    return f"Filtered data for date: {date} and incident type: {incident_type}"


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
        with gr.Tab("Carte"):
            gr.Markdown("### 🗺️ Carte Interactive")
            map_output = gr.Markdown()
            gr.Button("Afficher la carte").click(fn=display_map, inputs=[], outputs=map_output)

        with gr.Tab("Overview"):
            gr.Markdown("### 📊 Aperçu des données")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Filter Options")

                    # Filter 1: Date selection
                    gr.Markdown("##### Select Date")
                    year_input = gr.Dropdown(label="Year", choices=["2023", "2024", "2025"], value="2023")
                    month_input = gr.Dropdown(label="Month", choices=[str(i).zfill(2) for i in range(1, 13)],
                                              value="01")
                    day_input = gr.Dropdown(label="Day", choices=[str(i).zfill(2) for i in range(1, 32)], value="01")

                    # Filter 2: Incident type selection
                    gr.Markdown("##### Select Incident Type")
                    incident_type_input = gr.Dropdown(
                        label="Incident Type",
                        choices=["Type A", "Type B", "Type C"],
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

                with gr.Column():
                    gr.Markdown("#### Curve Display")
                    curve_1_output = gr.Markdown()
                    gr.Button("Display Curve 1").click(
                        fn=display_curve_1,
                        inputs=[],
                        outputs=curve_1_output
                    )

                    curve_2_output = gr.Markdown()
                    gr.Button("Display Curve 2").click(
                        fn=display_curve_2,
                        inputs=[],
                        outputs=curve_2_output
                    )

                    curve_3_output = gr.Markdown()
                    gr.Button("Display Curve 3").click(
                        fn=display_curve_3,
                        inputs=[],
                        outputs=curve_3_output
                    )

        with gr.Tab("Logs"):
                gr.Markdown("### 🗺️ Logs de la simulation")
                map_output = gr.Markdown()


app.launch()