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




__all__ = ['allocate_jobs']


def allocate_jobs(Ncpu, weights_in):

    """
    """

    weights_in = np.array(weights_in.ravel())

    weights = weights_in/weights_in.max()
    weights += weights.min()/2.0
    Nweight = weights.size
    ids = np.arange(Nweight)

    indices = np.argsort(weights)[::-1]
    jobs = list(zip(ids[indices], weights[indices]))
    Njob = len(jobs)

    loads_now = np.zeros(Ncpu, dtype=np.float64)
    workers = {iworker:[] for iworker in range(Ncpu)}

    while Njob > 0:

        job0 = jobs[0]
        id0  = job0[0]
        weight0 = job0[1]

        loads_diff = ((loads_now+weight0)-loads_now.min())**2
        iplace = np.argmin(loads_diff)

        loads_now[iplace] += weight0
        workers[iplace].append(id0)

        jobs.pop(0)
        Njob = len(jobs)

    for i in range(Ncpu):
        print(i, weights[workers[i]])
    print()

    Nround = max([len(workers[item]) for item in workers.keys()])
    weights_out = []
    i_next_round = np.arange(Ncpu)
    while Nround > 0:
        weight0_round = np.array([], dtype=np.float64)
        index0_round = np.array([], dtype=np.int32)
        for i in i_next_round:
            worker0 = workers[i]
            if len(worker0) > 0:
                index = worker0[0]

                weight0_round = np.append(weight0_round, weights[index])
                index0_round  = np.append(index0_round, i)

                # weights_out.append(weights_in[index])
                weights_out.append(weights[index])
                workers[i].pop(0)

        i_next_round = index0_round[np.argsort(weight0_round)]
        Nround = max([len(workers[item]) for item in workers.keys()])

        print(Nround)
        print(i_next_round)
        # print(workers)
        print(weights_out)

    sys.exit()

    weights_out = np.array(weights_out)
    print(weights_in.sum())
    print(weights_out.sum())

    return workers


if __name__ == '__main__':

    pass
