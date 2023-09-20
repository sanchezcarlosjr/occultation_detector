import gradio as gr
from occultation_detector.plotter import plot
from collections import OrderedDict
from occultation_detector.difractions import *

#Parametros basicos para el calculo
M=2**11 # Tamano de la malla en [px] 2048
lamb=600e-9 # Long de onda en [m]


#Parametros de la observacion (conocidos a Priori)
vE=29800 # velocidad de traslacion de la tierra  en m/s
vr=5000 #velocidad del cuerpo Pos si va en contra de la direccion de la tierra
ang=30 #angulo desde oposicion para calcular velocidad tangencial del objeto
fps=20 #frames por segundo
mV=15 # Magnitud aparente de la estrella # TODO: Magnitud 15, 12
nEst=1 #Seleccion de tipo espectral de estrella # TODO: 1,30 
nLamb=10 # Num de longitudes de onda a considerar para el calculo espectral spectra()
snr = SNR_TAOS2(mV) #Calcular senal a ruido segun TAOS-2

# TODO: Estrella tipo A0
#A0=1;A1=2;A2=3;A3=4;A4=5;A5=6;A7=7;F0=8;F2=9;F3=10;F5=11;F6=12;F7=13;F8=14
#G0=15;G1=16;G2=17;G5=18;G8=19;K0=20;K1=21;K2=22;K3=23;K4=24;K5=25;K7=26;
#M0=27;M1=28;M2=29;M3=30;M4=31;M5=32;M6=33;M7=34;M8=35

def simulate_lightcurve(d, ua, toffset, T, b):
    b = d*b
    D = calc_plano(d, lamb, ua) #Tamano del plano total en [m]
    O1 = pupilCO(M, D, d)  #Objeto 1: circular
    # CALCULAR PATRON CON CONTRIBUCION ESPECTRAL
    z = 1.496e11 * ua #Distancia del objeto en [m]
    I1 = fresnel(O1, M, D, z, lamb) #Patron 1 de difraccion monocromatico con fuente puntual
    I1s = spectra(O1, M, D, z, nEst, nLamb) #Esta funcion calcula el patron cromatico
    #CALCULAR PATRON PARA FUENTE EXTENDIDA
    tipo, R_star = calc_rstar(mV, nEst, ua) #Funcion para calcular el radio y tipo de la estrella usa estrellas.dat
    I1f = promedio_PD(I1s, R_star, D, M, d) #Funcion para calcular contribucion de fuente extendida
    #AGREGAR RUIDO DE POISSON
    I1n = add_ruido(I1f, mV) # Obtener patron de difraccion con ruido 
    #EXTRAER PERFIL DE DIFRACCION ojo T--> grados y b --> metros
    xc, yc = extraer_perfil(I1f, M, D, T, b) #Extraer perfil de difraccion sin ruido
    xb, yb = extraer_perfil(I1n, M, D, T, b) #Extraer perfil de difraccion con ruido
    #MUESTREAR SEGUN PARAMETROS CONOCIDOS DEFINIDOS AL PRINCIPIO
    x1, y1, x2, y2 = muestreos(yc, D, vr, fps, toff=toffset, vE=vE, opangle=0, ua=ua)  #fUNCION PARA MUESTREAR genera dos tuplas
    
    return (OrderedDict({
        "D": D,
        "z": z,
        "R_star": R_star,
        "tipo": tipo
    }), OrderedDict({
        "O1": O1,
        "I1": I1,
        "I1s": I1s,
        "I1f": I1f,
        "I1n": I1n,
        "xc": xc,
        "yc": yc,
        "xb": xb,
        "yb": yb,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }))

def update_plot(d, ua, toffset, T, b):
    response,series = simulate_lightcurve(d,ua,toffset,T,b)
    return plot(d, ua, toffset, T, b, series, response, nLamb, mV, snr)

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