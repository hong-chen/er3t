import er3t
import numpy as np



__all__ = ['pha_hg']



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

    """
    Henyey-Greenstein phase function object

    Input:
        asy_params: asymmetry parameters, Python list or numpy array, default=[-0.85, 0.85]
        angles: angles in degree, Python list or numpy array, default=np.array([0.0, 1.0, ..., 180.0])

    Output:
        phase function object

        pha0 = pha_hg(asy_params=[-0.85, 0.85], angles=np.linspace(0.0, 180.0, 1801))

        pha0.data['id']['data']: identification of the phase function object
        pha0.data['ang']['data']: angles in degree
        pha0.data['asy']['data']: asymmetry parameters in degree
        pha0.data['pha']['data']: phase functions
    """

    def __init__(self, asy_params=[-0.85, 0.85], angles=np.linspace(0.0, 180.0, 1801)):

        asy_params = np.array(asy_params)
        angles     = np.array(angles)

        pha = np.zeros((angles.size, asy_params.size), dtype=np.float64)

        for i, asy in enumerate(asy_params):
            pha[:, i] = cal_hg_pha_func(asy, angles)

        self.data = {
                'id' : {'data':'HG'      , 'name':'Henyey-Greenstein'  , 'unit':'N/A'},
                'ang': {'data':angles    , 'name':'Angle'              , 'unit':'degree'},
                'asy': {'data':asy_params, 'name':'Asymmetry parameter', 'unit':'N/A'},
                'pha': {'data':pha       , 'name':'Phase function'     , 'unit':'N/A'}
                }



if __name__ == '__main__':

    pass
