import os
import sys
import copy
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




__all__ = ['rearrange_jobs']


def rearrange_jobs(Ncpu, weights_in):

    """
    Purpose:
        Rearrange jobs to optimize computational time by assigning specific set of jobs
        to specific CPU so jobs from different CPUs can complete at a similar time length.

    Input:
        Ncpu: number of CPU
        weights_in: input weights, e.g., photons_dist

    Output:
        indices_out: optimized order for the input weights
    """

    # make input weights an array
    #/----------------------------------------------------------------------------\#
    weights_in = np.array(weights_in.ravel())
    #\----------------------------------------------------------------------------/#


    # weights
    #/----------------------------------------------------------------------------\#
    weights = weights_in.copy()
    weights += weights.min()
    Nweight = weights.size
    ids = np.arange(Nweight)
    #\----------------------------------------------------------------------------/#


    # sort weights in descending order
    #/----------------------------------------------------------------------------\#
    indices = np.argsort(weights)[::-1]
    ids     = ids[indices]
    weights = weights[indices]
    #\----------------------------------------------------------------------------/#


    # make jobs (python list)
    #/----------------------------------------------------------------------------\#
    jobs = list(zip(ids, weights))
    Njob = len(jobs)
    #\----------------------------------------------------------------------------/#


    # define workers (python dict)
    #/----------------------------------------------------------------------------\#
    workers = {iworker:[] for iworker in range(Ncpu)}
    #\----------------------------------------------------------------------------/#


    # assign jobs to workers in a way that the difference between the work load of
    # workers is minimized
    #/----------------------------------------------------------------------------\#
    loads_now = np.zeros(Ncpu, dtype=np.float64)
    while Njob > 0:

        job0 = jobs[0]

        id0  = job0[0]
        weight0 = job0[1]

        loads_diff = ((loads_now+weight0)-loads_now.min())**2
        iplace = np.argmin(loads_diff)

        loads_now[iplace] += weight0
        workers[iplace].append(job0)

        jobs.pop(0)
        Njob = len(jobs)
    #\----------------------------------------------------------------------------/#


    # sorted workers with number of jobs in descending order
    #/----------------------------------------------------------------------------\#
    rounds = [len(workers[i]) for i in range(Ncpu)]
    indices = np.argsort(np.array(rounds))[::-1]
    workers_sorted = {iworker:[] for iworker in range(Ncpu)}

    for i in range(Ncpu):
        workers_sorted[i] = workers[indices[i]]

    # workers_sorted_ = copy.deepcopy(workers_sorted) # variable backup
    #\----------------------------------------------------------------------------/#


    # rearrange jobs
    #/----------------------------------------------------------------------------\#
    rounds = [len(workers_sorted[i]) for i in range(Ncpu)]
    Nround = max(rounds)

    indices_out = []
    i_next_round = np.arange(Ncpu)

    counts = np.zeros(Ncpu)

    while Nround > 0:

        weight0_round = np.array([], dtype=np.float64)
        index0_round = np.array([], dtype=np.int32)

        for i in i_next_round:

            loads0 = workers_sorted[i]
            if len(loads0) > 0:
                id0 = loads0[0][0]
                weight0 = loads0[0][1]

                counts[i] += weight0

                weight0_round = np.append(weight0_round, weight0)
                index0_round  = np.append(index0_round, i)

                indices_out.append(id0)

                workers_sorted[i].pop(0)

        if Nround == max(rounds):
            weight0_base = weight0_round[:weight0_round.size].copy()
        else:
            weight0_base = weight0_base[:weight0_round.size].copy()
            weight0_base += weight0_round

        i_next_round = index0_round[np.argsort(weight0_base)]

        Nround = max(len(workers_sorted[i]) for i in range(Ncpu))

    indices_out = np.array(indices_out)
    #\----------------------------------------------------------------------------/#

    return indices_out


if __name__ == '__main__':

    pass
