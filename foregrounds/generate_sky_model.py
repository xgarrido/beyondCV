import pysm
from pysm.nominal import models
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import sys


nside = 2048
sky_config = {
            'synchrotron' : models("s1", nside),
            'dust' : models("d1", nside),
            'cmb' : models("c1", nside),
}

instrument_config = {
    'nside' : nside,
    'frequencies' : [93,100,143,145,217,225], #Expected in GHz
    'use_smoothing' : False,
    'beams' : np.ones(6) * 70., #Expected in arcmin
    'add_noise' : False,
    'sens_I' : np.ones(6), #Expected in units uK_RJ
    'sens_P' : np.ones(6),
    'noise_seed' : 1234,
    'use_bandpass' : False,
    'output_units' : 'uK_CMB',
    'output_directory' : 'sky/',
    'output_prefix' : 'sky',
}

mapDir = 'sky'
try:
    os.makedirs(mapDir)
except:
    pass

instrument = pysm.Instrument(instrument_config)
Sky = pysm.Sky(sky_config)
instrument.observe(Sky)
