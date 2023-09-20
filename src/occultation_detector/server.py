import gradio as gr
import matplotlib.pyplot as p
from occultation_detector.simulator import Simulator
from occultation_detector.plotter import plot

def update_plot(d, ua, toffset, T, b):
    observation_parameters = ObservationParameters()
    simulator = Simulator()
    response,series = simulator.simulate_circle_lightcurve(d,ua,toffset,T,b)
    return plot(d, ua, toffset, T, b, response, series, observation_parameters.nLamb, observation_parameters.mV, observation_parameters.snr)

def launch_web_server():
    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # TNO Simulator
        ## TAOS II
        """)
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                d_input = gr.Number(label="Diámetro del objeto en [m]", value=3000)
                ua_input = gr.Number(label="Distancia del objeto en unidades astronómicas", value=45)
                lamb_input = gr.Number(label="Wavelength", value=0)
                toffset_input = gr.Number(label="Offset en pixeles", value=0)
                T_input = gr.Slider(label="Dirección de lectura en grados", minimum=0, maximum=360, value=0)
                b_input = gr.Number(label="Parámetro de impacto en metros", value=0)
                btn = gr.Button("Run")
            with gr.Column(scale=2, min_width=600):
                outputs = [
                    *[gr.Plot(label="Plot") for _ in range(0,8)],
                    gr.JSON()
                ]
        btn.click(fn=update_plot, inputs=[d_input, ua_input, toffset_input, T_input, b_input], outputs=outputs)

    demo.launch(share=True)