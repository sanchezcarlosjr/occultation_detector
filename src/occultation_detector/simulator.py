import itertools
import pandas as pd
from collections import OrderedDict
from occulations_detector.difraciones import *
import numpy as np
from dataclasses import dataclass
import concurrent.futures
import os

@dataclass
class SimulationParameters:
    M: int = 2**11  # Size of the grid in [px] 2048
    lamb: float = 600e-9  # Wavelength in [m]

@dataclass
class ObservationParameters:
    vE: int = 29800  # Earth's translational velocity in m/s
    vr: int = 5000  # Body velocity. Positive if it goes against the Earth's direction
    ang: int = 30  # Angle from opposition to calculate tangential velocity of the object
    fps: int = 20  # Frames per second
    mV: int = 15  # Apparent magnitude of the star
    nEst: int = 1  # Selection of spectral type of star
    nLamb: int = 10  # Number of wavelengths to consider for the spectral calculation

    @property
    def snr(self):
        return SNR_TAOS2(self.mV)

@dataclass
class Simulator:
    sim_params: SimulationParameters
    obs_params: ObservationParameters
    def simulate_circle_lightcurve(self, d, ua, toffset, T, b):
        b = d * b
        D = calc_plano(d, self.sim_params.lamb, ua)
        O1 = pupilCO(self.sim_params.M, D, d)
        z = 1.496e11 * ua
        I1s = spectra(O1, self.sim_params.M, D, z, self.obs_params.nEst, self.obs_params.nLamb)
        tipo, R_star = calc_rstar(self.obs_params.mV, self.obs_params.nEst, ua)
        I1f = promedio_PD(I1s, R_star, D, self.sim_params.M, d)
        _, yc = extraer_perfil(I1f, self.sim_params.M, D, T, b)
        _, _, x2, y2 = muestreos(yc, D, self.obs_params.vr, self.obs_params.fps, toff=toffset, vE=self.obs_params.vE, opangle=0, ua=ua)
        
        return (OrderedDict({
            "D": D,
            "z": z,
            "R_star": R_star,
            "tipo": tipo
        }), {
            "x2": x2,
            "y2": y2,
        })

    def f(self, ordered_pair):
        response, series = self.simulate_circle_lightcurve(*ordered_pair)
        return [*ordered_pair, *list(response.values()), np.vstack((series['x2'],series['y2']))]

    def run(self):
        set_diameters = np.linspace(1000, 10000, 20)
        set_ua = np.linspace(40, 60, 5)
        set_toffset = [0]
        set_T = [0]
        set_b_impact_parameter = np.linspace(0, 3, 41)
        n = 20 * 5 * 41

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            print("Starting")
            future_to_value = {executor.submit(self.f, ordered_pair): i for i, ordered_pair in enumerate(itertools.product(set_diameters, set_ua, set_toffset, set_T, set_b_impact_parameter))}
            pairs = []
            index = 0
            for future in concurrent.futures.as_completed(future_to_value):
                i = future_to_value[future]
                index += 1
                print(f"{index*100/n:.2f}%    ", end="\r")
                try:
                    result = future.result()
                    np.save(f'results/taosii_circle_simulation_diffraction_profile/{self.obs_params.mV}_{self.obs_params.nEst}_{i}.npy', np.array(result[-1]))
                    pairs.append(result[0:-1])
                except Exception as exc:
                    print('%r generated an exception: %s' % (i, exc))
        
        pd.DataFrame(pairs, columns=['diameter', 'ua', 'toffset','T', 'b', "D", "z","R_star","tipo"]).to_csv('results/taosii_circle_simulation_features.csv')


