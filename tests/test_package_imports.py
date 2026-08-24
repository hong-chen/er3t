"""Fast package-boundary and import smoke tests."""

import importlib
import sys

import numpy as np


def test_top_level_import_is_lightweight():
    er3t = importlib.import_module("er3t")

    assert er3t.__version__
    assert "er3t.pre" not in sys.modules
    assert "er3t.rtm" not in sys.modules
    assert "er3t.util.modis" not in sys.modules


def test_public_subpackages_are_loaded_lazily():
    import er3t

    assert er3t.pre.__name__ == "er3t.pre"
    assert er3t.rtm.__name__ == "er3t.rtm"
    assert "er3t.rtm.shd" not in sys.modules


def test_numerical_utility_does_not_load_satellite_readers():
    from er3t.util import downscale

    source = np.arange(16).reshape(4, 4)
    result = downscale(source, (2, 2))

    np.testing.assert_allclose(result, [[2.5, 4.5], [10.5, 12.5]])
    assert "er3t.util.modis" not in sys.modules
    assert "er3t.util.viirs" not in sys.modules


def test_common_compatibility_exports_remain_available():
    import er3t

    assert er3t.params is er3t.common.params
    assert er3t.f_dtype is np.float32


def test_preprocessing_modules_with_corrected_imports_load():
    from er3t.pre.aer.aer_gen import aer_gen
    from er3t.pre.aer.aer_lasso import aer_lasso
    from er3t.pre.aer.aer_les import aer_les

    assert aer_gen.__name__ == "aer_gen"
    assert aer_lasso.__name__ == "aer_lasso"
    assert aer_les.__name__ == "aer_les"
