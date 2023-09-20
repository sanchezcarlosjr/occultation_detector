from dataclasses import dataclass
from collections import OrderedDict
from occultation_detector.difractions import *
from occultation_detector.plotter import plot

@dataclass
class ObservationParameters:
    vE: int = 29800  # Earth's translational velocity in m/s
    vr: int = 5000  # Body velocity. Positive if it goes against the Earth's direction
    ang: int = 30  # Angle from opposition to calculate tangential velocity of the object
    fps: int = 20  # Frames per second
    mV: int = 15  # Apparent magnitude of the star
    nEst: int = 1  # Selection of spectral type of star
    nLamb: int = 10  # Number of wavelengths to consider for the spectral calculation
    M: int = 2**11  # Size of the grid in [px] 2048
    lamb: float = 600e-9  # Wavelength in [m]

    @property
    def snr(self):
        return SNR_TAOS2(self.mV)


@dataclass
class Prediction:
    diameter: float
    ua: float
    toffset: float
    T: float
    b: float
    D: float  # TODO: remove of the model
    z: float # TODO: remove of the model
    R_star: float # TODO: remove of the model
    type: float  # TODO: remove of the model
    observation_parameters: ObservationParameters = ObservationParameters()
    
    def plot(self):
        response, series = self.find_key_features()
        plot(self.diameter, self.ua, self.toffset, self.T, self.b, series, response, self.observation_parameters.nLamb, self.observation_parameters.mV, self.observation_parameters.snr)

    def find_key_features(self):
        ob = self.observation_parameters
        D = calc_plano(self.diameter, ob.lamb, self.ua)
        O1 = pupilCO(ob.M, D, self.diameter)
        z = 1.496e11 * self.ua
        I1 = fresnel(O1, ob.M, D, z, ob.lamb)
        I1s = spectra(O1, ob.M, D, z, ob.nEst, ob.nLamb)
        tipo, R_star = calc_rstar(ob.mV, ob.nEst, self.ua)
        I1f = promedio_PD(I1s, R_star, D, ob.M, self.diameter)
        I1n = add_ruido(I1f, ob.mV)
        xc, yc = extraer_perfil(I1f, ob.M, D, self.T, self.b)
        xb, yb = extraer_perfil(I1n, ob.M, D, self.T, self.b)

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
            "yb": yb
        }))
