import os

__all__ = ['fdir_data']

fdir_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
fdir_data_abs = os.path.join(fdir_data, 'abs')
fdir_data_pha = os.path.join(fdir_data, 'pha')
fdir_data_atmmod = os.path.join(fdir_data, 'atmmod')
