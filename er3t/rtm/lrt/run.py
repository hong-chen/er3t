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
            f.write('%-20s %s\n' % (key, init.input_dict[key]))

    if init.input_dict_extra is not None:
        for key in init.input_dict_extra.keys():
            if key not in init.mute_list:
                f.write('%-20s %s\n' % (key, init.input_dict_extra[key]))

    if verbose:
        f.write('verbose')
    else:
        f.write('quiet')

    f.close()

    # Run libRadtran "$ uvspec < input.txt > output.txt"
    os.system('%s < %s > %s' % (init.executable_file, init.input_file, init.output_file))



def lrt_run_mp(inits, ncpu=6):

    """
    Use multiprocessing to run lrt_run with multiple CPUs

    Input:
        Python list of lrt_init objects
    """

    pool = mp.Pool(processes=ncpu)
    pool.outputs = pool.map(lrt_run, inits)
    pool.close()
    pool.join()



if __name__ == '__main__':

    pass
