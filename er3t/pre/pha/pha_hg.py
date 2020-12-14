import er3t

class pha_mca:

    fdir = '%s/pha' % er3t.common.fdir_data

    def __init__(self):

        pass

    def make_h_g(self):

        """
        Henyey-Greenstein phase function
        """

        self.value = 0.85

        pass

    def make_16g(self):

        pass

if __name__ == '__main__':

    a = pha_mca()
    print(a.fdir)

    pass

