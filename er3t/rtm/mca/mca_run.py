import os
import sys
import datetime
import multiprocessing as mp
import numpy as np
import er3t.common

try:
    from tqdm import tqdm
except ImportError:
    msg = 'Warning [mca_run.py]: To use \'mca_run.py\', \'tqdm\' needs to be installed.'
    print(msg)



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
                 fname_sh   = None,   \
                 verbose    = True,   \
                 quiet      = False   \
                ):

        if executable is None:
            if er3t.common.has_mcarats:
                executable = os.environ['MCARATS_V010_EXE']
            else:
                sys.exit('Error   [er3t.rtm.mca.mca_run]: Cannot locate MCARaTS. Please make sure MCARaTS is installed and specified at enviroment variable <MCARaTS_V010_EXE>.')

        Nfile = len(fnames_inp)
        if len(fnames_out) != Nfile:
            sys.exit('Error   [mca_run]: Inconsistent input and output files.')

        self.Ncpu    = Ncpu
        self.quiet   = quiet
        self.verbose = verbose

        if not isinstance(photons, np.ndarray):
            photons_dist = np.repeat(photons, Nfile)
        else:
            if photons.size == Nfile:
                photons_dist = photons.copy()
            else:
                sys.exit('Error   [mca_run]: Cannot distribute photon set of %d over %d runs.' % (photons.size, Nfile))

        mp_mode = mp_mode.lower()
        if mp_mode in ['mpi', 'openmpi']:
            mp_mode = 'mpi'
        elif mp_mode in ['python', 'multiprocessing', 'py', 'mp', 'pymp']:
            mp_mode = 'py'
        elif mp_mode in ['batch', 'shell', 'bash', 'hpc']:
            mp_mode = 'sh'
        else:
            sys.exit('Error   [mca_run]: Cannot understand input \'mp_mode=\'%s\'\'' % mp_mode)
        self.mp_mode = mp_mode


        self.commands = []
        for i in range(Nfile):

            input_file  = os.path.abspath(fnames_inp[i])
            output_file = os.path.abspath(fnames_out[i])

            fdir_out    = os.path.dirname(output_file)
            if not os.path.exists(fdir_out):
                os.system('mkdir -p %s' % fdir_out)

            if (Ncpu > 1) and (self.mp_mode == 'mpi'):
                command = 'mpirun -n %d %s %d %d %s %s' % (Ncpu, executable, photons_dist[i], solver, input_file, output_file)
            else:
                command = '%s %d %d %s %s' % (executable, photons_dist[i], solver, input_file, output_file)

            self.commands.append(command)


        if self.mp_mode == 'mpi' or self.mp_mode == 'py':
            self.run()
        if self.mp_mode == 'sh':
            self.save(fname=fname_sh)


    def run(self):

        if self.mp_mode == 'mpi':

            with tqdm(total=len(self.commands)) as pbar:

                for command in self.commands:
                    execute_command(command)
                    pbar.update(1)

        elif self.mp_mode == 'py':

            with mp.Pool(processes=self.Ncpu) as pool:
                r = list(tqdm(pool.imap(execute_command, self.commands), total=len(self.commands)))


    def save(self, fname=None):

        if fname is None:
            fname = 'mca_batch_script_%s.sh' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if not self.quiet:
            print('Message [mca_run]: Creating batch script \'%s\' ...' % fname)

        with open(fname, 'w') as f:

            for command in self.commands:
                f.write(command + '\n')

        os.system('chmod +x %s' % fname)



def execute_command(command, verbose=False):

    if verbose:
        print('Message [mca_run]: Executing \'%s\' ...' % command)
    os.system(command)



if __name__ == '__main__':

    pass
