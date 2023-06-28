import warnings

import er3t



class benchmark_ipa:

    """
    under development

    intend to benchmark MCARaTS with libRadtran for IPA (also known as 1D) calculations
    """

    def __init__(self):

        self.flux_clear_sky()
        self.flux_cloud()
        self.flux_aerosol()
        self.flux_cloud_and_aerosol()

        pass

    def flux_clear_sky(self):

        pass

    def flux_cloud(self):

        pass

    def flux_aerosol(self):

        pass


    def flux_cloud_and_aerosol(self):

        pass



class benchmark_3d:

    """
    under development:

    intend to benchmark 3d calculations using I3RC cases
    """

    def __init__(self):

        pass



if __name__ == '__main__':

    warnings.warn('\nCaution in use: under development ...\n')

    bm_ipa = benchmark_ipa()

    bm_3d  = benchmark_3d()
