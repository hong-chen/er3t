import er3t



def cal_hg_pha_func(asy, ang):

    """
    Henyey-Greenstein phase function

    Input:
        asy: asymmetry parameter
        ang: angles in degree (within the range of -180 to 180)

    Output:
        pha: phase function
    """


    mu = np.cos(np.deg2rad(ang))
    pha = 0.5*(1.0-asy**2.0)/((1.0-2.0*asy*mu+asy**2.0)**1.5)

    return pha



class pha_hg:

    def __init__(self, asy_params, angles):

        pha = np.zeros((angles.size, asy_params.size), dtype=np.float64)

        for i, asy in enumerate(asy_params):
            pha[:, i] = cal_hg_pha_func(asy, angles)

        self.data = {
                'ang': angles,
                'asy': asy_params,
                'pha': pha
                }



if __name__ == '__main__':

    pass
