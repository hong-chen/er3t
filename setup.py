import os
import sys
from setuptools import setup, find_packages

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
     name = 'er3t',
     version = '0.0.1',
     description = 'Education and Research 3D Radiative Transfer Toolbox',
     long_description = long_description,
     classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU GPLv3 License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
     keywords = '3D radiation radiative transfer',
     url = 'https://github.com/hong-chen/er3t',
     author = 'Sebastian Schmidt, Hong Chen',
     author_email = 'sebastian.schmidt@lasp.colorado.edu, hong.chen.cu@gmail.com',
     license = 'GNU GPLv3',
     packages = find_packages(),
     install_requires = ['nose', 'numpy', 'scipy', 'h5py'],
     python_requires = '~=3.7',
     include_package_data = True,
     zip_safe = False
     )
