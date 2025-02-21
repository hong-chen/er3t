import os
import multiprocessing as mp



__all__ = ['lrt_run', 'lrt_run_mp']



def lrt_run(init, verbose=False):

    """
    Create libRadtran input file that contains RTM parameters.

    Input:
        lrt_init object
    """

    f = open(init.input_file, 'w')
    for key in init.input_dict.keys():
        if key not in init.mute_list:
            content = '%-20s %s\n' % (key, init.input_dict[key])
            f.write(content)

    if init.input_dict_extra is not None:
        for key in init.input_dict_extra.keys():
            content = '%-20s %s\n' % (key, init.input_dict_extra[key])
            f.write(content)

    if verbose:
        f.write('verbose')
    else:
        f.write('quiet')

    f.close()

    # Run libRadtran "$ uvspec < input.txt > output.txt"
    os.system('%s < %s > %s' % (init.executable_file, init.input_file, init.output_file))



def lrt_run_mp(inits, Ncpu=None):

    """
    Use multiprocessing to run lrt_run with multiple CPUs

    Input:
        Python list of lrt_init objects
    """

    if Ncpu is None:
        Ncpu = mp.cpu_count() - 1

    try:
        from tqdm import tqdm

        print('\nMessage [lrt_run_mp]: running libRadtran ...')
        with mp.Pool(processes=Ncpu) as pool:
            r = list(tqdm(pool.imap_unordered(lrt_run, inits), total=len(inits)))

    except ImportError:

        pool = mp.Pool(processes=Ncpu)
        pool.outputs = pool.map_async(lrt_run, inits)
        pool.close()
        pool.join()



if __name__ == '__main__':

    pass
