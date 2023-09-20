import pytest

from occultation_detector.pipeline import transform

__author__ = "sanchezcarlosjr"
__copyright__ = "sanchezcarlosjr"
__license__ = "MIT"


def test_transform():
    assert transform([0,1,2,3,4,5,6,7,8]) == Prediction(diameter=0, ua=1, toffset=2, T=3, b=4, D=5, z=6, R_star=7, type=8, observation_parameters=ObservationParameters(vE=29800, vr=5000, ang=30, fps=20, mV=15, nEst=1, nLamb=10, M=2048, lamb=6e-07))]