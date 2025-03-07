import os
import sys
from setuptools import setup, find_packages

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
     name = 'er3t',
     version = '0.2.0',
     description = 'Education and Research 3D Radiative Transfer Toolbox',
     long_description = long_description,
     classifiers = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU GPLv3 License',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
     keywords = '3D radiation radiative transfer',
     url = 'https://github.com/hong-chen/er3t',
     author = 'Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Hong Chen, Sebastian Schmidt',
     author_email = 'vikas.hanasogenataraja@lasp.colorado.edu,\
                     yu-wen.chen@colorado.edu,\
                     ken.hirata@colorado.edu,\
                     hong.chen@lasp.colorado.edu,\
                     sebastian.schmidt@lasp.colorado.edu',
     license = 'GNU GPLv3',
     packages = find_packages(),
     install_requires = [
         'requests',
         'tqdm',
         'gdown',
         'numpy',
         'scipy',
         'pyhdf',
         'h5py',
         'netCDF4',
         'owslib',
         'cartopy'],
     python_requires = '~=3.12',
     scripts = ['bin/lss', 'bin/lsa', 'bin/sdown'],
     include_package_data = True,
     zip_safe = False
     )
