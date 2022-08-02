import os
import sys

__all__ = ['fdir_data']

has_mcarats    = ('MCARATS_V010_EXE' in dict(os.environ))
has_libradtran = ('LIBRADTRAN_V2_DIR' in dict(os.environ))

fdir_data        = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
fdir_data_abs    = os.path.join(fdir_data, 'abs')
fdir_data_pha    = os.path.join(fdir_data, 'pha')
fdir_data_atmmod = os.path.join(fdir_data, 'atmmod')
fdir_data_slit   = os.path.join(fdir_data, 'slit')
fdir_data_ssfr   = os.path.join(fdir_data_slit, 'ssfr')
