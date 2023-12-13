import itertools

from joblib import Parallel, delayed, dump
from joblib_progress import joblib_progress

from occultation_detector.difractions import *
import numpy as np
from dataclasses import dataclass
from occultation_detector.prediction import ObservationParameters
import concurrent.futures
import os


@dataclass
class Simulator:
    obs_params: ObservationParameters = ObservationParameters()

    def simulate_circle_lightcurve(self, d, ua, toffset, T, b):
        D = calc_plano(d, self.obs_params.lamb, ua)
        O1 = pupilCO(self.obs_params.M, D, d)
        z = 1.496e11 * ua
        I1s = spectra(O1, self.obs_params.M, D, z, self.obs_params.nEst, self.obs_params.nLamb)
        tipo, R_star = calc_rstar(self.obs_params.mV, self.obs_params.nEst, ua)
        I1f = promedio_PD(I1s, R_star, D, self.obs_params.M, d)
        _, yc = extraer_perfil(I1f, self.obs_params.M, D, T, b)
        _, _, x2, y2 = muestreos(yc, D, self.obs_params.vr, self.obs_params.fps, toff=toffset, vE=self.obs_params.vE,
                                 opangle=0, ua=ua)

        return (
            [y2[:90]],
            [d, ua, toffset, T, b]
        )

    def run(self):
        set_diameters = np.linspace(1000, 10000, 10000)
        set_ua = np.linspace(40, 60, 10000)
        set_toffset = [0]
        set_T = [0]
        set_b_impact_parameter = np.linspace(0, 3, 42)
        features = itertools.product(set_diameters, set_ua, set_toffset, set_T, set_b_impact_parameter)
        timeseries = []
        labels = []
        for result in Parallel(n_jobs=-1, verbose=1, return_as='generator')(delayed(self.simulate_circle_lightcurve)(*instance) for instance in features):
            timeseries.append(result[0])
            labels.append(result[1])
        filename = os.path.join(os.getcwd(), 'taosii_circle_simulation_diffraction_profile.joblib')
        dump([np.array(timeseries), np.array(labels)], filename)
