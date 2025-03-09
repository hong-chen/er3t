#!/usr/bin/env python

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path to import from modis module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from er3t.util.modis import (
    modis_03, modis_l1b, modis_l2, modis_35_l2, modis_mvcm_cldmsk_l2,
    modis_04, modis_09, modis_09a1, modis_43a1, modis_43a3, modis_tiff
)

class TestModisReaders(unittest.TestCase):
    """Test case for MODIS data reader classes"""

    @patch('er3t.util.modis.SD')
    def test_modis_03(self, mock_sd):
        """Test modis_03 class for reading geolocation data"""
        print("\nTesting modis_03 reader...")

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Mock dataset selections
        mock_lat = MagicMock()
        mock_lon = MagicMock()
        mock_sza = MagicMock()
        mock_saa = MagicMock()
        mock_vza = MagicMock()
        mock_vaa = MagicMock()

        # Mock return values
        mock_lat.return_value = np.zeros((10, 10))
        mock_lon.return_value = np.zeros((10, 10))
        mock_sza.return_value = np.zeros((10, 10))
        mock_saa.return_value = np.zeros((10, 10))
        mock_vza.return_value = np.zeros((10, 10))
        mock_vaa.return_value = np.zeros((10, 10))

        # Setup return values for mock selects
        mock_sd_instance.select.side_effect = lambda name: {
            'Latitude': mock_lat,
            'Longitude': mock_lon,
            'SolarZenith': mock_sza,
            'SolarAzimuth': mock_saa,
            'SensorZenith': mock_vza,
            'SensorAzimuth': mock_vaa
        }[name]

        # Test modis_03
        try:
            reader = modis_03(fnames=['mock_modis03.hdf'])
            print("modis_03 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_03 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    def test_modis_l1b(self, mock_sd):
        """Test modis_l1b class for reading Level 1B data"""
        print("\nTesting modis_l1b reader...")

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Create mock MODIS_03 object with required structure
        mock_modis03 = MagicMock()
        mock_modis03.data = {
            'lon': {'data': np.zeros(100)},
            'lat': {'data': np.zeros(100)},
        }
        mock_modis03.logic = {'mock_modis03.hdf': {'1km': np.ones(100, dtype=bool)}}

        # Setup mock attributes and methods
        filename = 'MOD021KM.hdf'
        mock_band_1km = MagicMock()
        mock_band_1km.attributes.return_value = {'radiance_scales': [1.0], 'radiance_offsets': [0.0],
                                                'reflectance_scales': [1.0], 'reflectance_offsets': [0.0],
                                                'corrected_counts_scales': [1.0], 'corrected_counts_offsets': [0.0]}
        mock_sd_instance.select.return_value = mock_band_1km

        # Test modis_l1b
        try:
            reader = modis_l1b(fnames=[filename], f03=mock_modis03, bands=[1])
            print("modis_l1b reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_l1b initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    def test_modis_l2(self, mock_sd):
        """Test modis_l2 class for reading Level 2 cloud product"""
        print("\nTesting modis_l2 reader...")

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Mock datasets
        mock_lat = MagicMock()
        mock_lon = MagicMock()
        mock_ctp = MagicMock()
        mock_cot = MagicMock()
        mock_cer = MagicMock()
        mock_cwp = MagicMock()
        mock_cot_pcl = MagicMock()
        mock_cer_pcl = MagicMock()
        mock_cwp_pcl = MagicMock()
        mock_cot_err = MagicMock()
        mock_cer_err = MagicMock()
        mock_cwp_err = MagicMock()

        # Mock return values
        mock_lat.attributes.return_value = {'actual_range': [-90, 90]}
        mock_lon.attributes.return_value = {'actual_range': [-180, 180]}

        # Setup return values for mock selects
        mock_sd_instance.select.side_effect = lambda name: {
            'Latitude': mock_lat,
            'Longitude': mock_lon,
            'Cloud_Phase_Optical_Properties': mock_ctp,
            'Cloud_Optical_Thickness': mock_cot,
            'Cloud_Effective_Radius': mock_cer,
            'Cloud_Water_Path': mock_cwp,
            'Cloud_Optical_Thickness_PCL': mock_cot_pcl,
            'Cloud_Effective_Radius_PCL': mock_cer_pcl,
            'Cloud_Water_Path_PCL': mock_cwp_pcl,
            'Cloud_Optical_Thickness_Uncertainty': mock_cot_err,
            'Cloud_Effective_Radius_Uncertainty': mock_cer_err,
            'Cloud_Water_Path_Uncertainty': mock_cwp_err
        }.get(name, MagicMock())

        # Test modis_l2
        try:
            reader = modis_l2(fnames=['mock_modis06.hdf'])
            print("modis_l2 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_l2 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    def test_modis_35_l2(self, mock_sd):
        """Test modis_35_l2 class for reading Level 2 cloud mask product"""
        print("\nTesting modis_35_l2 reader...")

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Mock datasets
        mock_lat = MagicMock()
        mock_lon = MagicMock()
        mock_cld_msk = MagicMock()
        mock_qa = MagicMock()

        # Mock return values
        mock_lat.attributes.return_value = {'actual_range': [-90, 90]}
        mock_lon.attributes.return_value = {'actual_range': [-180, 180]}

        # Setup return values for mock selects
        mock_sd_instance.select.side_effect = lambda name: {
            'Latitude': mock_lat,
            'Longitude': mock_lon,
            'Cloud_Mask': mock_cld_msk,
            'Quality_Assurance': mock_qa
        }.get(name, MagicMock())

        # Test modis_35_l2
        try:
            reader = modis_35_l2(fnames=['mock_mod35.hdf'])
            print("modis_35_l2 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_35_l2 initialization failed with error: {e}")

    @patch('er3t.util.modis.Dataset')
    def test_modis_mvcm_cldmsk_l2(self, mock_dataset):
        """Test modis_mvcm_cldmsk_l2 class for reading MVCM cloud mask product"""
        print("\nTesting modis_mvcm_cldmsk_l2 reader...")

        # Setup mock Dataset object
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        # Mock variables
        mock_clear_sky = MagicMock()
        mock_cloud_mask = MagicMock()

        # Mock dictionary access for mock_dataset_instance
        mock_dataset_instance.__getitem__.side_effect = lambda name: {
            'geophysical_data/Clear_Sky_Confidence': mock_clear_sky,
            'geophysical_data/Integer_Cloud_Mask': mock_cloud_mask
        }[name]

        # Test modis_mvcm_cldmsk_l2
        try:
            with patch('er3t.util.modis.get_data_nc', return_value=np.zeros((10, 10))):
                reader = modis_mvcm_cldmsk_l2(fname='mock_cldmsk_l2.nc')
                print("modis_mvcm_cldmsk_l2 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_mvcm_cldmsk_l2 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    def test_modis_04(self, mock_sd):
        """Test modis_04 class for reading deep blue aerosol data"""
        print("\nTesting modis_04 reader...")

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Mock datasets
        mock_lat = MagicMock()
        mock_lon = MagicMock()
        mock_aod = MagicMock()
        mock_ae = MagicMock()
        mock_at = MagicMock()
        mock_acf = MagicMock()
        mock_ssa = MagicMock()

        # Mock return values
        mock_lat.attributes.return_value = {'actual_range': [-90, 90]}
        mock_lon.attributes.return_value = {'actual_range': [-180, 180]}

        # Setup return values for mock selects
        mock_sd_instance.select.side_effect = lambda name: {
            'Latitude': mock_lat,
            'Longitude': mock_lon,
            'Deep_Blue_Aerosol_Optical_Depth_550_Land': mock_aod,
            'Deep_Blue_Angstrom_Exponent_Land': mock_ae,
            'Aerosol_Type_Land': mock_at,
            'Aerosol_Cloud_Fraction_Land': mock_acf,
            'Deep_Blue_Spectral_Single_Scattering_Albedo_Land': mock_ssa
        }[name]

        # Test modis_04
        try:
            reader = modis_04(fnames=['mock_mod04.hdf'])
            print("modis_04 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_04 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    def test_modis_09(self, mock_sd):
        """Test modis_09 class for reading surface reflectance data"""
        print("\nTesting modis_09 reader...")

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Mock datasets
        mock_lat = MagicMock()
        mock_lon = MagicMock()
        mock_reflectance = MagicMock()

        # Mock return values
        mock_lat.attributes.return_value = {'actual_range': [-90, 90]}
        mock_lon.attributes.return_value = {'actual_range': [-180, 180]}

        # Mock datasets() to return list of datasets
        mock_sd_instance.datasets.return_value = {
            '1km Surface Reflectance Band 1': mock_reflectance
        }

        # Setup return values for mock selects
        mock_sd_instance.select.side_effect = lambda name: {
            'Latitude': mock_lat,
            'Longitude': mock_lon,
            '1km Surface Reflectance Band 1': mock_reflectance
        }[name]

        # Test modis_09
        try:
            with patch('er3t.util.modis.get_data_h4', return_value=np.zeros((10, 10))):
                reader = modis_09(fname='mock_mod09.hdf')
                print("modis_09 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_09 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    @patch('er3t.util.modis.ccrs')
    def test_modis_09a1(self, mock_ccrs, mock_sd):
        """Test modis_09a1 class for reading 8-day surface reflectance"""
        print("\nTesting modis_09a1 reader...")

        # Mock projections
        mock_sinusoidal = MagicMock()
        mock_platecarree = MagicMock()
        mock_ccrs.Sinusoidal.MODIS = mock_sinusoidal
        mock_ccrs.PlateCarree.return_value = mock_platecarree

        # Setup transform points method
        mock_platecarree.transform_points.return_value = np.zeros((10, 10, 3))

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Test modis_09a1
        try:
            with patch('er3t.util.modis.cal_sinusoidal_grid',
                      return_value=(np.linspace(-20000000, 20000000, 37),
                                   np.linspace(10000000, -10000000, 19))):
                with patch('er3t.util.modis.get_data_h4', return_value=np.zeros((10, 10))):
                    reader = modis_09a1(fnames=['MOD09A1.A2020001.h10v10.061.hdf'])
                    print("modis_09a1 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_09a1 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    @patch('er3t.util.modis.ccrs')
    def test_modis_43a1(self, mock_ccrs, mock_sd):
        """Test modis_43a1 class for reading surface BRDF data"""
        print("\nTesting modis_43a1 reader...")

        # Mock projections
        mock_sinusoidal = MagicMock()
        mock_platecarree = MagicMock()
        mock_ccrs.Sinusoidal.MODIS = mock_sinusoidal
        mock_ccrs.PlateCarree.return_value = mock_platecarree

        # Setup transform points method
        mock_platecarree.transform_points.return_value = np.zeros((10, 10, 3))

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Test modis_43a1
        try:
            with patch('er3t.util.modis.cal_sinusoidal_grid',
                      return_value=(np.linspace(-20000000, 20000000, 37),
                                   np.linspace(10000000, -10000000, 19))):
                with patch('er3t.util.modis.get_data_h4', return_value=np.zeros((10, 10, 3))):
                    reader = modis_43a1(fnames=['MCD43A1.A2020001.h10v10.061.hdf'])
                    print("modis_43a1 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_43a1 initialization failed with error: {e}")

    @patch('er3t.util.modis.SD')
    @patch('er3t.util.modis.ccrs')
    def test_modis_43a3(self, mock_ccrs, mock_sd):
        """Test modis_43a3 class for reading surface albedo data"""
        print("\nTesting modis_43a3 reader...")

        # Mock projections
        mock_sinusoidal = MagicMock()
        mock_platecarree = MagicMock()
        mock_ccrs.Sinusoidal.MODIS = mock_sinusoidal
        mock_ccrs.PlateCarree.return_value = mock_platecarree

        # Setup transform points method
        mock_platecarree.transform_points.return_value = np.zeros((10, 10, 3))

        # Setup mock SD object
        mock_sd_instance = MagicMock()
        mock_sd.return_value = mock_sd_instance

        # Test modis_43a3
        try:
            with patch('er3t.util.modis.cal_sinusoidal_grid',
                      return_value=(np.linspace(-20000000, 20000000, 37),
                                   np.linspace(10000000, -10000000, 19))):
                with patch('er3t.util.modis.get_data_h4', return_value=np.zeros((10, 10))):
                    reader = modis_43a3(fnames=['MCD43A3.A2020001.h10v10.061.hdf'])
                    print("modis_43a3 reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_43a3 initialization failed with error: {e}")

    @patch('er3t.util.modis.gdal')
    def test_modis_tiff(self, mock_gdal):
        """Test modis_tiff class for reading GeoTIFF files"""
        print("\nTesting modis_tiff reader...")

        # Setup mock gdal object
        mock_gdal_instance = MagicMock()
        mock_gdal.Open.return_value = mock_gdal_instance

        # Mock data reading
        mock_gdal_instance.ReadAsArray.return_value = np.zeros((3, 10, 10))
        mock_gdal_instance.RasterXSize = 10
        mock_gdal_instance.RasterYSize = 10
        mock_gdal_instance.GetGeoTransform.return_value = (0, 1, 0, 0, 0, 1)

        # Test modis_tiff
        try:
            reader = modis_tiff(fname='mock_modis.tiff')
            print("modis_tiff reader initialized successfully")
        except Exception as e:
            self.fail(f"modis_tiff initialization failed with error: {e}")

if __name__ == "__main__":
    print("===== Testing MODIS Reader Classes =====")
    unittest.main(verbosity=1)
