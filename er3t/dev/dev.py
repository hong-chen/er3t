import os
import sys
import shutil
import datetime
import time
import requests
import urllib.request
from io import StringIO
import numpy as np
from scipy import interpolate, stats
import warnings

import er3t




__all__ = ['photon_dist']


def photon_dist(Nphoton, weights, base_rate=0.1):

    Ndist = weights.size
    photons_dist = np.int_(Nphoton*(1.0-base_rate)*weights) + np.int_(Nphoton*base_rate/Ndist)

    Ndiff = Nphoton - photons_dist.sum()

    if Ndiff >= 0:
        photons_dist[np.argmin(weights)] += Ndiff
    else:
        photons_dist[np.argmax(weights)] += Ndiff


    # photon distribution over gs of correlated-k
    #/----------------------------------------------------------------------------\#
    # photons_min_ipa0 = int(self.Nx*self.Ny*100)
    # photons_min_3d0  = min(int(1e7), photons_min_ipa0)

    # if weights is None:
    #     self.np_mode = 'evenly'
    #     weights = np.repeat(1.0/self.Ng, Ng)
    # else:
    #     self.np_mode = 'weighted'

    # photons_dist = np.int_(photons*weights)
    # Ndiff        = (photons_dist.sum()-photons)
    # index        = np.argmax(photons_dist)
    # photons_dist[index] = photons_dist[index] - Ndiff

    # if ((photons_dist<photons_min_ipa0).sum() > 0) and (self.solver=='IPA'):
    #     photons_dist += (abs(photons_min_ipa0-photons_dist.min()))
    # elif ((photons_dist<photons_min_3d0).sum() > 0) and (self.solver=='3D'):
    #     photons_dist += (abs(photons_min_3d0-photons_dist.min()))

    # self.photons = np.tile(photons_dist, Nrun)
    # self.photons_per_set = photons_dist.sum()
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    pass
