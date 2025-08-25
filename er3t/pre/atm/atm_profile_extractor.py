"""
Atmospheric Profile Extractor

This module provides tools for extracting and processing atmospheric profile data from
various data sets and saves them to different file formats appropriate for ingestion into
atmospheric models (ARCSIXAtmMod).
"""

import os
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from er3t.util import constants
from er3t.util.plot_util import set_plot_fonts, MPL_STYLE_PATH


def cams_eac4(fname, outdir, date='20240605_1500', extent=[-65, -47.5, 83.3, 84]):
    """
    Extract and process atmospheric profile data from CAMS EAC4 dataset.
    This function computes regional mean values for atmospheric variables including ozone, nitrogen
    dioxide, specific humidity, and temperature.

    doi: doi.org/10.24381/d58bbf47

    Args:
    ----
        fname : str
            Path to the CAMS EAC4 NetCDF file to be processed.
        outdir : str
            Output directory where the converted file will be saved.
        date : str, optional
            Date and time string in format 'YYYYMMDD_HHMM' (default: '20240605_1500').
            Used to select the specific time slice from the dataset.
        extent : list of float, optional
            Geographic bounding box as [lon_min, lon_max, lat_min, lat_max] in degrees
            (default: [-65, -47.5, 83.3, 84]). Defines the region for spatial filtering.
    Returns:
    -------
        tuple
            A tuple containing:
            - pressure : xarray.DataArray
                Pressure levels from the dataset
            - ozone_mean : numpy.ma.MaskedArray
                Regional mean ozone mixing ratio profile at given time
            - no2_mean : numpy.ma.MaskedArray
                Regional mean nitrogen dioxide mixing ratio profile at given time
            - q_mean : numpy.ma.MaskedArray
                Regional mean specific humidity profile at given time
            - t_mean : numpy.ma.MaskedArray
                Regional mean temperature profile at given time
    """
    # open file
    fnc = xr.open_dataset(fname)

    # select desired time slice
    dt = datetime.datetime.strptime(date, '%Y%m%d_%H%M')
    date_str = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

    fnc = fnc.sel(valid_time=date_str)

    # create regional filter to select data within the specified extent
    region_filter = (fnc.longitude >= extent[0]) & (fnc.longitude <= extent[1]) & (fnc.latitude >= extent[2]) & (fnc.latitude <= extent[3])
    fnc = fnc.where(region_filter, drop=True)

    # pressure
    pressure = fnc['pressure_level'][:].to_masked_array()
    print(f"Found {pressure.size} pressure levels: {pressure} {fnc['pressure_level'].attrs['units']}")

    # geopotential
    geopotential_mean = fnc['z'].mean(dim=["longitude", "latitude"]).to_masked_array()
    geopotential_std = fnc['z'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Geopotential Coefficient of Variation (%): {np.round(geopotential_std*100/geopotential_mean, 1)}")

    # divide geopotential by g to get z
    altitude = geopotential_mean / constants.g

    # ozone
    ozone_mean = fnc['go3'].mean(dim=["longitude", "latitude"]).to_masked_array()
    ozone_std = fnc['go3'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Ozone Coefficient of Variation (%): {np.round(ozone_std*100/ozone_mean, 1)}")

    # nitrogen dioxide
    no2_mean = fnc['no2'].mean(dim=["longitude", "latitude"]).to_masked_array()
    no2_std = fnc['no2'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"NO2 Coefficient of Variation (%): {np.round(no2_std*100/no2_mean, 1)}")

    # specific humidity
    q_mean = fnc['q'].mean(dim=["longitude", "latitude"]).to_masked_array()
    q_std = fnc['q'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Specific humidity Coefficient of Variation (%): {np.round(q_std*100/q_mean, 1)}")

    # temperature
    t_mean = fnc['t'].mean(dim=["longitude", "latitude"]).to_masked_array()
    t_std = fnc['t'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Temperature Coefficient of Variation (%): {np.round(t_std*100/t_mean, 1)}")

    print('Ozone', fnc['go3'].attrs['long_name'], fnc['go3'].attrs['units'])
    print('Nitrogen Dioxide', fnc['no2'].attrs['long_name'], fnc['no2'].attrs['units'])
    print('Specific Humidity', fnc['q'].attrs['long_name'], fnc['q'].attrs['units'])
    print('Temperature', fnc['t'].attrs['long_name'], fnc['t'].attrs['units'])
    print('Altitude', 'Altitude derived from geopotential height', 'm')

    assert pressure.size == ozone_mean.size == no2_mean.size == q_mean.size == t_mean.size == altitude.size, "Pressure and variable sizes do not match. Found the following sizes pressure: {pressure.size}, ozone: {ozone_mean.size}, no2: {no2_mean.size}, q: {q_mean.size}, t: {t_mean.size}, altitude: {altitude.size}"

    # save data and units to csv file and dat file
    output_csv_file = os.path.basename(fname).replace('.nc', f'_{date}.csv')
    output_dat_file = os.path.basename(fname).replace('.nc', f'_{date}.dat')
    output_csv_fname = os.path.join(outdir, output_csv_file)
    output_dat_fname = os.path.join(outdir, output_dat_file)

    with open(output_csv_fname, 'w') as f:
        f.write("pressure,z,o3,no2,q,temperature\n")
        for i in range(pressure.size):
            f.write(f"{pressure[i]},{altitude[i]},{ozone_mean[i]},{no2_mean[i]},{q_mean[i]},{t_mean[i]}\n")

        print(f"Processed data from {os.path.basename(fname)} saved to {output_csv_fname}")

    with open(output_dat_fname, 'w') as f:
        f.write("pressure,z,o3,no2,q,temperature\n")
        f.write(f"{fnc['pressure_level'].attrs['units']},m,{fnc['go3'].attrs['units']},{fnc['no2'].attrs['units']},{fnc['q'].attrs['units']},{fnc['t'].attrs['units']} \n")
        for i in range(pressure.size):
            f.write(f"{pressure[i]},{altitude[i]},{ozone_mean[i]},{no2_mean[i]},{q_mean[i]},{t_mean[i]}\n")

        print(f"Processed data from {os.path.basename(fname)} saved to {output_dat_fname}")

    plot_preprocessed_data(output_csv_fname, outdir=outdir)

    return pressure, altitude, ozone_mean, no2_mean, q_mean, t_mean


def cams_egg4(fname, outdir, date='20240605_1500', extent=[-65, -47.5, 83.3, 84]):
    """
    Extract and process greenhouse gas data from CAMS EGG4 dataset.
    This function computes regional mean values for greenhouse gas variables including co2, ch4,
    specific humidity, and temperature.

    doi: doi.org/10.24381/cda4ed31

    Args:
    ----
        fname : str
            Path to the CAMS EGG4 NetCDF file to be processed.
        outdir : str
            Output directory where the converted file will be saved.
        date : str, optional
            Date and time string in format 'YYYYMMDD_HHMM' (default: '20240605_1500').
            Used to select the specific time slice from the dataset.
        extent : list of float, optional
            Geographic bounding box as [lon_min, lon_max, lat_min, lat_max] in degrees
            (default: [-65, -47.5, 83.3, 84]). Defines the region for spatial filtering.
    Returns:
    -------
        tuple
            A tuple containing:
            - pressure : xarray.DataArray
                Pressure levels from the dataset
            - ch4_mean : numpy.ma.MaskedArray
                Regional mean methane mixing ratio profile at given time
            - co2_mean : numpy.ma.MaskedArray
                Regional mean carbon dioxide mixing ratio profile at given time
            - q_mean : numpy.ma.MaskedArray
                Regional mean specific humidity profile at given time
            - t_mean : numpy.ma.MaskedArray
                Regional mean temperature profile at given time
    """
    # open the data set
    fnc = xr.open_dataset(fname)

    # select the specific time slice
    dt = datetime.datetime.strptime(date, '%Y%m%d_%H%M')
    date_str = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

    fnc = fnc.sel(valid_time=date_str)

    # create regional filter to reduce size
    region_filter = (fnc.longitude >= extent[0]) & (fnc.longitude <= extent[1]) & (fnc.latitude >= extent[2]) & (fnc.latitude <= extent[3])
    try:
        fnc = fnc.where(region_filter, drop=True)
    except ValueError:
        raise ValueError("No data found in the specified extent. Please check the extent and try again.")

    pressure = fnc['pressure_level'][:].to_masked_array()
    print(f"Found {pressure.size} pressure levels: {pressure} {fnc['pressure_level'].attrs['units']}")

    # geopotential height
    geopotential_mean = fnc['z'].mean(dim=["longitude", "latitude"]).to_masked_array()
    geopotential_std = fnc['z'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Geopotential Coefficient of Variation (%): {np.round(geopotential_std*100/geopotential_mean, 1)}")

    # get z from gph
    altitude = geopotential_mean / constants.g

    # methane
    ch4_mean = fnc['ch4'].mean(dim=["longitude", "latitude"]).to_masked_array()
    ch4_std = fnc['ch4'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Methane Coefficient of Variation (%): {np.round(ch4_std*100/ch4_mean, 3)}")

    # co2
    co2_mean = fnc['co2'].mean(dim=["longitude", "latitude"]).to_masked_array()
    co2_std = fnc['co2'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"CO2 Coefficient of Variation (%): {np.round(co2_std*100/co2_mean, 3)}")

    # specific humidity
    q_mean = fnc['q'].mean(dim=["longitude", "latitude"]).to_masked_array()
    q_std = fnc['q'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Specific humidity Coefficient of Variation (%): {np.round(q_std*100/q_mean, 3)}")

    # temperature
    t_mean = fnc['t'].mean(dim=["longitude", "latitude"]).to_masked_array()
    t_std = fnc['t'].std(dim=["longitude", "latitude"]).to_masked_array()
    print(f"Temperature Coefficient of Variation (%): {np.round(t_std*100/t_mean, 3)}")

    print('Methane', fnc['ch4'].attrs['long_name'], fnc['ch4'].attrs['units'])
    print('CO2', fnc['co2'].attrs['long_name'], fnc['co2'].attrs['units'])
    print('Specific Humidity', fnc['q'].attrs['long_name'], fnc['q'].attrs['units'])
    print('Temperature', fnc['t'].attrs['long_name'], fnc['t'].attrs['units'])
    print('Altitude', 'Altitude derived from geopotential height', 'm')

    assert pressure.size == ch4_mean.size == co2_mean.size == q_mean.size == t_mean.size == altitude.size, "Pressure and variable sizes do not match. Found the following sizes pressure: {pressure.size}, methane: {ch4_mean.size}, co2: {co2_mean.size}, q: {q_mean.size}, t: {t_mean.size}, altitude: {altitude.size}"

    # save data and units to csv file and dat file
    output_csv_file = os.path.basename(fname).replace('.nc', f'_{date}.csv')
    output_dat_file = os.path.basename(fname).replace('.nc', f'_{date}.dat')
    output_csv_fname = os.path.join(outdir, output_csv_file)
    output_dat_fname = os.path.join(outdir, output_dat_file)

    with open(output_csv_fname, 'w') as f:
        f.write("pressure,z,ch4,co2,q,temperature\n")
        for i in range(pressure.size):
            f.write(f"{pressure[i]},{altitude[i]},{ch4_mean[i]},{co2_mean[i]},{q_mean[i]},{t_mean[i]}\n")

        print(f"Processed data from {os.path.basename(fname)} saved to {output_csv_fname}")

    with open(output_dat_fname, 'w') as f:
        f.write("pressure,z,ch4,co2,q,temperature\n")
        f.write(f"{fnc['pressure_level'].attrs['units']},m,{fnc['ch4'].attrs['units']},{fnc['co2'].attrs['units']},{fnc['q'].attrs['units']},{fnc['t'].attrs['units']}\n")
        for i in range(pressure.size):
            f.write(f"{pressure[i]},{altitude[i]},{ch4_mean[i]},{co2_mean[i]},{q_mean[i]},{t_mean[i]}\n")

        print(f"Processed data from {os.path.basename(fname)} saved to {output_dat_fname}")

    plot_preprocessed_data(output_csv_fname, outdir=outdir)

    return pressure, altitude, ch4_mean, co2_mean, q_mean, t_mean


def plot_preprocessed_data(csv_fname, outdir):
    """
    Create vertical profile plots of atmospheric data from CSV file.

    Args:
    ----
        csv_fname : str
            Path to CSV file containing pressure and atmospheric variable data
        outdir : str
            Directory to save the output plots
    """
    # Read the CSV data
    df = pd.read_csv(csv_fname)

    # get all plottable columns except pressure
    plottable_columns = df.columns[df.columns != 'pressure']

    fig, axes = plt.subplots(1, len(plottable_columns), figsize=(20, 8))
    set_plot_fonts(plt)
    plt.style.use(MPL_STYLE_PATH)

    for i, col in enumerate(plottable_columns):
        axes[i].plot(df[col], df['pressure'])
        axes[i].set_xlabel(f'{col}')
        axes[i].set_ylabel('Pressure (hPa)')
        axes[i].set_title(f'{col} Profile')
        axes[i].invert_yaxis()
        axes[i].grid(True, alpha=0.3)

    # add main title
    fig.suptitle(f'Vertical Profiles of Atmospheric Variables for {os.path.basename(csv_fname)}')

    plt.tight_layout()

    # Save the plot
    plot_fname = os.path.basename(csv_fname).replace('.csv', '_profiles.png')
    full_fname = os.path.join(outdir, plot_fname)
    fig.savefig(full_fname, dpi=300, bbox_inches='tight')
    print(f"Vertical profile plots saved to {full_fname}")

    plt.close()




if __name__ == "__main__":

    from er3t.common import fdir_er3t
    # Define the input and output file paths
    cams_eac4_file = os.path.join(fdir_er3t, 'data/test_data/cams_eac4_o3_no2_q/data_plev.nc')

    # create outdir
    outdir = os.path.join(fdir_er3t, 'data/preprocessed_data_atm_corr/')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    date_str = '20240605_1500' # june 5 at 1500Z
    extent = [-65, -47.5, 83.3, 84]

    pressure, z, ozone_mean, no2_mean, q_mean, t_mean = cams_eac4(cams_eac4_file, outdir=outdir, date=date_str, extent=extent)

    # Define the input and output file paths
    cams_egg4_file = os.path.join(fdir_er3t, 'data/test_data/cams_egg4_ch4_co2/data_plev.nc')

    # date_str = '20240605_1500' # june 5 at 1500Z
    date_str = '20190605_1500' # 2019 data since 2024 data is not available
    extent = [-65, -47.5, 83.3, 84]

    pressure, z, ch4_mean, co2_mean, q_mean, t_mean = cams_egg4(cams_egg4_file, outdir=outdir, date=date_str, extent=extent)
