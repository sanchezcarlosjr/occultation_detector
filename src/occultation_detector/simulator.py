import itertools

from joblib import Parallel, delayed, dump
from joblib_progress import joblib_progress

from occultation_detector.difractions import *
import numpy as np
from dataclasses import dataclass
from occultation_detector.prediction import ObservationParameters
import concurrent.futures
import os

import h5py
import numpy as np
import uuid


#Parametros basicos para el calculo
M=2**11 # Tamano de la malla en [px] 2048
lamb=600e-9 # Long de onda en [m]


#Parametros de la observacion (conocidos a Priori)
vE=29800 # velocidad de traslacion de la tierra  en m/s
vr=5000 #velocidad del cuerpo Pos si va en contra de la direccion de la tierra
ang=30 #angulo desde oposicion para calcular velocidad tangencial del objeto
fps=20 #frames por segundo
mV=12 # Magnitud aparente de la estrella
nEst=30 #Seleccion de tipo espectral de estrella
#A0=1;A1=2;A2=3;A3=4;A4=5;A5=6;A7=7;F0=8;F2=9;F3=10;F5=11;F6=12;F7=13;F8=14
#G0=15;G1=16;G2=17;G5=18;G8=19;K0=20;K1=21;K2=22;K3=23;K4=24;K5=25;K7=26;
#M0=27;M1=28;M2=29;M3=30;M4=31;M5=32;M6=33;M7=34;M8=35
nLamb=1 # Num de longitudes de onda a considerar para el calculo espectral spectra()


def save_to_hdf5(data, filename):
    """
    Save the output of simulate_circle_lightcurve to an HDF5 file.
    
    Parameters:
        data (tuple): Output of simulate_circle_lightcurve.
        filename (str): HDF5 file name.
    """
    with h5py.File(filename, 'w') as h5file:
        for key, value in data.items():
            # Save scalars or arrays
            if hasattr(value, '__len__') and not isinstance(value, str):
                h5file.create_dataset(key, data=value)
            else:
                h5file.create_dataset(key, data=value)

@dataclass
class Simulator:
    def simulate_circle_lightcurve(self, diameter, astronomical_unit, time_offset, angle_in_degrees, baseline):
        total_plane_size = calc_plano(diameter, lamb, astronomical_unit)  # Size of the total plane in meters
        circular_pupil = pupilCO(M, total_plane_size, diameter)  # Circular pupil object
    
        # CALCULATE PATTERN WITH SPECTRAL CONTRIBUTION
        object_distance = 1.496e11 * astronomical_unit  # Distance of the object in meters
        monochromatic_diffraction_pattern = fresnel(circular_pupil, M, total_plane_size, object_distance, lamb)
        chromatic_diffraction_pattern = spectra(circular_pupil, M, total_plane_size, object_distance, nEst, nLamb)
    
        # CALCULATE PATTERN FOR EXTENDED SOURCE
        star_type, star_radius = calc_rstar(mV, nEst, astronomical_unit)  # Calculate the star's radius and type using stars.dat
        extended_source_pattern = promedio_PD(chromatic_diffraction_pattern, star_radius, total_plane_size, M, diameter)
    
        # ADD POISSON NOISE
        noisy_diffraction_pattern = add_ruido(extended_source_pattern, mV)
    
        # EXTRACT DIFFRACTION PROFILE
        # `angle_in_degrees` represents angular positions; `baseline` represents baseline distance in meters
        profile_without_noise_x, profile_without_noise_y = extraer_perfil(extended_source_pattern, M, total_plane_size, angle_in_degrees, baseline)
        profile_with_noise_x, profile_with_noise_y = extraer_perfil(noisy_diffraction_pattern, M, total_plane_size, angle_in_degrees, baseline)
    
        # SAMPLE BASED ON PREDEFINED PARAMETERS
        sampled_x1, sampled_y1, sampled_x2, sampled_y2 = muestreos(
            profile_without_noise_y, total_plane_size, vr, fps,
            toff=time_offset, vE=vE, opangle=0, ua=astronomical_unit
        )
    
        return {
             "total_plane_size": total_plane_size,
            "object_distance": object_distance,
            "star_radius": star_radius,
            "star_type": star_type,
            "total_plane_size": total_plane_size,
            "object_distance": object_distance,
            "star_radius": star_radius,
            "star_type": star_type,
            "circular_pupil": circular_pupil,
            "monochromatic_diffraction_pattern": monochromatic_diffraction_pattern,
            "chromatic_diffraction_pattern": chromatic_diffraction_pattern,
            "extended_source_pattern": extended_source_pattern,
            "noisy_diffraction_pattern": noisy_diffraction_pattern,
            "profile_without_noise_x": profile_without_noise_x,
            "profile_without_noise_y": profile_without_noise_y,
            "profile_with_noise_x": profile_with_noise_x,
            "profile_with_noise_y": profile_with_noise_y,
            "sampled_x1": sampled_x1,
            "sampled_y1": sampled_y1,
            "sampled_x2": sampled_x2,
            "sampled_y2": sampled_y2,
        }

    def simulate_contact_binaries_lightcurve(self, diameter, astronomical_unit, time_offset, angle_in_degrees, baseline):
        total_plane_size = calc_plano(diameter, lamb, astronomical_unit)  # Size of the total plane in meters
        circular_pupil = pupil_doble(M, total_plane_size, diameter)  # Circular pupil object
    
        # CALCULATE PATTERN WITH SPECTRAL CONTRIBUTION
        object_distance = 1.496e11 * astronomical_unit  # Distance of the object in meters
        monochromatic_diffraction_pattern = fresnel(circular_pupil, M, total_plane_size, object_distance, lamb)
        chromatic_diffraction_pattern = spectra(circular_pupil, M, total_plane_size, object_distance, nEst, nLamb)
    
        # CALCULATE PATTERN FOR EXTENDED SOURCE
        star_type, star_radius = calc_rstar(mV, nEst, astronomical_unit)  # Calculate the star's radius and type using stars.dat
        extended_source_pattern = promedio_PD(chromatic_diffraction_pattern, star_radius, total_plane_size, M, diameter)
    
        # ADD POISSON NOISE
        noisy_diffraction_pattern = add_ruido(extended_source_pattern, mV)
    
        # EXTRACT DIFFRACTION PROFILE
        # `angle_in_degrees` represents angular positions; `baseline` represents baseline distance in meters
        profile_without_noise_x, profile_without_noise_y = extraer_perfil(extended_source_pattern, M, total_plane_size, angle_in_degrees, baseline)
        profile_with_noise_x, profile_with_noise_y = extraer_perfil(noisy_diffraction_pattern, M, total_plane_size, angle_in_degrees, baseline)
    
        # SAMPLE BASED ON PREDEFINED PARAMETERS
        sampled_x1, sampled_y1, sampled_x2, sampled_y2 = muestreos(
            profile_without_noise_y, total_plane_size, vr, fps,
            toff=time_offset, vE=vE, opangle=0, ua=astronomical_unit
        )
    
        return {
             "total_plane_size": total_plane_size,
            "object_distance": object_distance,
            "star_radius": star_radius,
            "star_type": star_type,
            "total_plane_size": total_plane_size,
            "object_distance": object_distance,
            "star_radius": star_radius,
            "star_type": star_type,
            "circular_pupil": circular_pupil,
            "monochromatic_diffraction_pattern": monochromatic_diffraction_pattern,
            "chromatic_diffraction_pattern": chromatic_diffraction_pattern,
            "extended_source_pattern": extended_source_pattern,
            "noisy_diffraction_pattern": noisy_diffraction_pattern,
            "profile_without_noise_x": profile_without_noise_x,
            "profile_without_noise_y": profile_without_noise_y,
            "profile_with_noise_x": profile_with_noise_x,
            "profile_with_noise_y": profile_with_noise_y,
            "sampled_x1": sampled_x1,
            "sampled_y1": sampled_y1,
            "sampled_x2": sampled_x2,
            "sampled_y2": sampled_y2,
        }

    def run(self):
        set_diameters = np.linspace(1000, 10000, 1000)
        set_ua = np.linspace(40, 45, 1)
        set_toffset = [0]
        set_T = [0]
        set_b_impact_parameter = np.linspace(0, 0, 42)
        features = itertools.product(set_diameters, set_ua, set_toffset, set_T, set_b_impact_parameter)
        for result in Parallel(n_jobs=-1, verbose=1, return_as='generator')(delayed(self.simulate_contact_binaries_lightcurve)(*instance) for instance in features):
            save_to_hdf5(result, f'results/contact_binaries/'+ str(uuid.uuid4()) + '.h5')

