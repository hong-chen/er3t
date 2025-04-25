import os
import sys
import time
import datetime
import warnings
import multiprocessing as mp
import numpy as np

import er3t.common



__all__ = ['shd_run']



class shd_run:

    """
    Run SHDOM

    Input:
        fnames_inp: positional argument, Python list that contains file path (string type) of the SHDOM input files
        executable=: keyword argument, string, path of the executable file of SHDOM, default is system variable 'SHDOM_EXE'

        solver=  : keyword argument, integer, 0: 3d mode, 1: partial-3d mode, 2: ipa mode, default=0
        Ncpu=    : keyword argument, integer, number of CPUs to run, default=1
        mp_mode= : keyword argument, string type, can be 'py' for Python multiprocessing, 'mpi' for openMPI, 'sh' for batch mode, default='py'
        fname_sh=: keyword argument, string type, if on batch mode, this variable is used to specify the file path of the batch file, default=None
        verbose= : keyword argument, boolen type, verbose tag, defaut=False
        quiet=   : keyword argument, boolen type, quiet tag, defaut=False

    Output:
        1. If mp_mode='py' or mp_mode='mpi', command line argument will be executed and SHDOM will create corresponding output files

        2. if mp_mode='sh', a shell script will be saved.
    """

    def __init__(self,
                 fnames_inp, \
                 executable = None, \
                 Ncpu       = 1,      \
                 mp_mode    = 'mpi',   \
                 has_mpi    = er3t.common.has_mpi, \
                 fname_sh   = None,   \
                 verbose    = er3t.common.params['verbose'],   \
                 quiet      = False   \
                ):

        if executable is None:
            if er3t.common.has_shdom:
                executable = os.environ['SHDOM_EXE']
            else:
                msg = '\nError [shd_run]: Cannot locate SHDOM. Please make sure SHDOM is installed and specified at enviroment variable <SHDOM_EXE>.'
                raise OSError(msg)

        Nfile = len(fnames_inp)

        self.Ncpu    = Ncpu
        self.quiet   = quiet
        self.verbose = verbose

        mp_mode = mp_mode.lower()
        if mp_mode in ['mpi', 'openmpi']:
            mp_mode = 'mpi'
        elif mp_mode in ['python', 'multiprocessing', 'py', 'mp', 'pymp']:
            mp_mode = 'py'
        elif mp_mode in ['batch', 'shell', 'bash', 'hpc', 'sh']:
            mp_mode = 'sh'
        else:
            msg = '\nError [shd_run]: Cannot understand input <mp_mode=\'%s\'>.' % mp_mode
            raise OSError(msg)

        self.mp_mode = mp_mode

        self.commands = []

        indices = np.arange(Nfile)

        for i in indices:

            input_file  = os.path.abspath(fnames_inp[i])

            fdir_out    = os.path.dirname(input_file)
            if not os.path.exists(fdir_out):
                os.system('mkdir -p %s' % fdir_out)

            if (Ncpu > 1) and has_mpi:
                command = 'mpirun -np %d %s %s' % (Ncpu, executable, input_file)
            else:
                command = '%s %s' % (executable, input_file)

            self.commands.append(command)

        if self.mp_mode == 'mpi' or self.mp_mode == 'py':

            self.run()

        if self.mp_mode == 'sh':

            self.save(fname=fname_sh)


    def run(self):

        if self.verbose:
            for command in self.commands:
                print('Message [shd_run]: Executing <%s> ...' % command)

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
            fname = 'er3t-shd_shell-script_%18.7f.sh' % time.time()

        if not self.quiet:
            print('Message [shd_run]: Creating batch script <%s> ...' % fname)

        with open(fname, 'w') as f:

            for command in self.commands:
                f.write(command + '\n')

        os.system('chmod +x %s' % fname)



def execute_command(command):

    os.system(command)



if __name__ == '__main__':

    pass
