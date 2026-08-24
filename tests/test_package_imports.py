"""Fast package-boundary and import smoke tests."""

import importlib
import inspect
import io
import logging
import sys

import numpy as np
import pytest


def test_top_level_import_is_lightweight():
    er3t = importlib.import_module("er3t")

    assert er3t.__version__
    assert "er3t.pre" not in sys.modules
    assert "er3t.rtm" not in sys.modules
    assert "er3t.sat.readers.modis" not in sys.modules


def test_public_subpackages_are_loaded_lazily():
    import er3t

    assert er3t.pre.__name__ == "er3t.pre"
    assert er3t.rtm.__name__ == "er3t.rtm"
    assert "er3t.rtm.shd" not in sys.modules


def test_numerical_utility_does_not_load_satellite_readers():
    from er3t.core import downscale

    source = np.arange(16).reshape(4, 4)
    result = downscale(source, (2, 2))

    np.testing.assert_allclose(result, [[2.5, 4.5], [10.5, 12.5]])
    assert "er3t.sat.readers.modis" not in sys.modules
    assert "er3t.sat.readers.viirs" not in sys.modules


def test_common_is_an_explicit_submodule():
    import er3t

    assert er3t.common.f_dtype is np.float32
    with pytest.raises(AttributeError):
        _ = er3t.f_dtype


def test_preprocessing_modules_with_corrected_imports_load():
    from er3t.pre.aer.aer_gen import aer_gen
    from er3t.pre.aer.aer_lasso import aer_lasso
    from er3t.pre.aer.aer_les import aer_les

    assert aer_gen.__name__ == "aer_gen"
    assert aer_lasso.__name__ == "aer_lasso"
    assert aer_les.__name__ == "aer_les"


def test_core_settings_and_resources_are_explicit():
    from er3t.core import default_settings, resource_path

    first = default_settings()
    second = default_settings()
    first.Ncpu = 1

    assert second.Ncpu != first.Ncpu
    assert resource_path("atmmod", "afglus.dat", must_exist=True).is_file()


def test_new_satellite_and_io_boundaries_are_public():
    from er3t.io import load_h5
    from er3t.sat.products import get_product_catalog
    from er3t.sat.readers.modis import modis_l1b

    catalog = get_product_catalog()
    assert callable(load_h5)
    assert modis_l1b.__name__ == "modis_l1b"
    assert "MOD03" in catalog


def test_visualization_api_has_no_legacy_switch():
    import er3t.visualization as visualization

    assert "legacy" not in visualization.__all__


def test_public_classes_use_concrete_bases_and_safe_defaults():
    from er3t.core._logger import Ear3tLogger
    from er3t.rtm.mca.mcarats import mcarats_ng
    from er3t.sat.readers.modis import modis_03
    from er3t.visualization.plot import PreprocessFigure

    assert issubclass(Ear3tLogger, logging.Logger)
    assert PreprocessFigure.__bases__ == (object,)
    assert inspect.signature(mcarats_ng.init_atm).parameters["atm_1ds"].default is None
    assert inspect.signature(modis_03.read_vars).parameters["vnames"].default is None


def test_cli_dialogue_separator_uses_requested_width():
    from er3t.cli._output import dialogue_separator, print_dialogue

    stream = io.StringIO()
    print_dialogue("results", stream=stream)

    assert dialogue_separator(width=7) == "─" * 7
    assert stream.getvalue().splitlines()[1] == "results"


def test_package_logger_mirrors_messages_to_an_optional_file(tmp_path):
    import er3t.common
    from er3t.core import configure_logging, start_log_session

    log_file = tmp_path / "er3t.log"
    configure_logging(log_file=log_file)
    start_log_session("pre/atm", width=9)
    er3t.common.logger.info("saved message")
    er3t.common.configure_logging()

    contents = log_file.read_text(encoding="utf-8")
    assert "saved message" in contents
    assert "─" * 9 in contents
    assert "pre/atm" in contents
    assert contents.splitlines()[:3] == ["─" * 9, "pre/atm", "─" * 9]
    assert "\x1b" not in contents
