# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
import astropy.units as u
from astropy import cosmology
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
import pandas as pd
import time
from sympy.physics.wigner import wigner_3j
from pysz import *

def verifyOoM(result, desired, req_order=1):
    # Check if result is within desired order-of-magnitude of true value
    if isinstance(result, u.Quantity) and isinstance(desired, u.Quantity):
        # Strip units from result and desired to allow for log10
        result = result.to(desired.unit).value
        desired = desired.value

    order_of_mag = abs(np.log10(result) - np.log10(desired)) 
    return np.all(order_of_mag <= req_order)

def getWigner3j(l1: u.Quantity, l2: u.Quantity, l3: u.Quantity):
    w3j = np.empty(len(l1))
    for i in range(len(l1)):
        w3j[i] = wigner_3j(l1[i]*u.rad, l2[i]*u.rad, l3[i]*u.rad, 0, 0, 0)
    return w3j

__all__ = ['os', 'np', 'unittest', 'u', 'cosmology', 'Planck18', 'plt',
           'csv', 'mp', 'pd', 'time', 'verifyOoM', 'getWigner3j',
           'calculators', 'distributions', 'integrators', 'structures']