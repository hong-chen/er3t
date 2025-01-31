import os
import sys
import time
import datetime
import warnings
import multiprocessing as mp
import numpy as np

import er3t.common



__all__ = ['mca_run']



class mca_run:

    """
    Run MCARaTS

    Input:
        fnames_inp: positional argument, Python list that contains file path (string type) of the MCARaTS input files
        fnames_out: positional argument, Python list that contains file path (string type) of the MCARaTS output files
        executable=: keyword argument, string, path of the executable file of MCARaTS, default is system variable 'MCARATS_EXE'

         photons= : keyword argument, integer, number of photons, default=1e6
         solver=  : keyword argument, integer, 0: 3d mode, 1: partial-3d mode, 2: ipa mode, default=0
         Ncpu=    : keyword argument, integer, number of CPUs to run, default=1
         mp_mode= : keyword argument, string type, can be 'py' for Python multiprocessing, 'mpi' for openMPI, 'sh' for batch mode, default='py'
         fname_sh=: keyword argument, string type, if on batch mode, this variable is used to specify the file path of the batch file, default=None
         verbose= : keyword argument, boolen type, verbose tag, defaut=False
         quiet=   : keyword argument, boolen type, quiet tag, defaut=False

    Output:
        1. If mp_mode='py' or mp_mode='mpi', command line argument will be executed and MCARaTS will create corresponding output files

        2. if mp_mode='sh', a shell script will be saved.
    """

    def __init__(self,
                 fnames_inp, \
                 fnames_out, \
                 executable = None, \
                 photons    = 1.0e6,  \
                 solver     = 0,      \
                 Ncpu       = 1,      \
                 mp_mode    = 'py',   \
                 optimize   = True,   \
                 has_mpi    = er3t.common.has_mpi, \
                 fname_sh   = None,   \
                 verbose    = er3t.common.params['verbose'],   \
                 quiet      = False   \
                ):

        if executable is None:
            if er3t.common.has_mcarats:
                executable = os.environ['MCARATS_V010_EXE']
            else:
                msg = '\nError [mca_run]: Cannot locate MCARaTS. Please make sure MCARaTS is installed and specified at enviroment variable <MCARaTS_V010_EXE>.'
                raise OSError(msg)

        Nfile = len(fnames_inp)
        if len(fnames_out) != Nfile:
            msg = '\nError [mca_run]: Inconsistent input and output files.'
            raise OSError(msg)

        self.Ncpu    = Ncpu
        self.quiet   = quiet
        self.verbose = verbose

        if not isinstance(photons, np.ndarray):
            photons_dist = np.repeat(photons, Nfile)
        else:
            if photons.size == Nfile:
                photons_dist = photons.copy()
            else:
                msg = 'Error [mca_run]: Cannot distribute photon set of %d over %d runs.' % (photons.size, Nfile)
                raise ValueError(msg)

        mp_mode = mp_mode.lower()
        if mp_mode in ['mpi', 'openmpi']:
            mp_mode = 'mpi'
            has_mpi = True
        elif mp_mode in ['python', 'multiprocessing', 'py', 'mp', 'pymp']:
            mp_mode = 'py'
        elif mp_mode in ['batch', 'shell', 'bash', 'hpc', 'sh']:
            mp_mode = 'sh'
        else:
            msg = '\nError [mca_run]: Cannot understand input <mp_mode=\'%s\'>.' % mp_mode
            raise OSError(msg)
        self.mp_mode = mp_mode

        self.commands = []

        if Ncpu > 1 and optimize:
            indices = rearrange_jobs(Ncpu, photons_dist)
        else:
            indices = np.arange(Nfile)

        for i in indices:

            input_file  = os.path.abspath(fnames_inp[i])
            output_file = os.path.abspath(fnames_out[i])

            fdir_out    = os.path.dirname(output_file)
            if not os.path.exists(fdir_out):
                os.system('mkdir -p %s' % fdir_out)

            if (Ncpu > 1) and has_mpi:
                command = 'mpirun -n %d %s %d %d %s %s' % (Ncpu, executable, photons_dist[i], solver, input_file, output_file)
            else:
                command = '%s %d %d %s %s' % (executable, photons_dist[i], solver, input_file, output_file)

            self.commands.append(command)

        if self.mp_mode == 'mpi' or self.mp_mode == 'py':
            self.run()
        if self.mp_mode == 'sh':
            self.save(fname=fname_sh)


    def run(self):

        if self.verbose:
            for command in self.commands:
                print('Message [mca_run]: Executing <%s> ...' % command)

        if self.mp_mode == 'mpi':

            try:
                from tqdm import tqdm

                with tqdm(total=len(self.commands)) as pbar:
                    for command in self.commands:
                        execute_command(command)
                        pbar.update(1)

            except ImportError:

                for command in self.commands:
                    execute_command(command)

        elif self.mp_mode == 'py':

            try:
                from tqdm import tqdm

                with mp.Pool(processes=self.Ncpu) as pool:
                    r = list(tqdm(pool.imap(execute_command, self.commands), total=len(self.commands)))
                    pool.close()
                    pool.join()

            except ImportError:

                with mp.Pool(processes=self.Ncpu) as pool:
                    pool.outputs = pool.map(execute_command, self.commands)
                    pool.close()
                    pool.join()


    def save(self, fname=None):

        if fname is None:
            fname = 'er3t-mca_shell-script_%18.7f.sh' % time.time()

        if not self.quiet:
            print('Message [mca_run]: Creating batch script <%s> ...' % fname)

        with open(fname, 'w') as f:

            for command in self.commands:
                f.write(command + '\n')

        os.system('chmod +x %s' % fname)



def execute_command(command):

    os.system(command)



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
    #╭────────────────────────────────────────────────────────────────────────────╮#
    weights_in = np.array(weights_in.ravel())
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # weights
    #╭────────────────────────────────────────────────────────────────────────────╮#
    weights = weights_in.copy()
    weights += weights.min()
    Nweight = weights.size
    ids = np.arange(Nweight)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # sort weights in descending order
    #╭────────────────────────────────────────────────────────────────────────────╮#
    indices = np.argsort(weights)[::-1]
    ids     = ids[indices]
    weights = weights[indices]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # make jobs (python list)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    jobs = list(zip(ids, weights))
    Njob = len(jobs)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define workers (python dict)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    workers = {iworker:[] for iworker in range(Ncpu)}
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # assign jobs to workers in a way that the difference between the work load of
    # workers is minimized
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # sorted workers with number of jobs in descending order
    #╭────────────────────────────────────────────────────────────────────────────╮#
    rounds = [len(workers[i]) for i in range(Ncpu)]
    indices = np.argsort(np.array(rounds))[::-1]
    workers_sorted = {iworker:[] for iworker in range(Ncpu)}

    for i in range(Ncpu):
        workers_sorted[i] = workers[indices[i]]

    # workers_sorted_ = copy.deepcopy(workers_sorted) # variable backup
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # rearrange jobs
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return indices_out



if __name__ == '__main__':

    pass
