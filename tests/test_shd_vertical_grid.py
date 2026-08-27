from types import SimpleNamespace

import numpy as np

from er3t.rtm.shd._vertical import (
    convert_propgen_to_layer_level,
    merged_level_grid,
    remap_cloud_layers,
    temperatures_on_grid,
)
from er3t.rtm.shd.shd_atm import shd_atm_1d
from er3t.rtm.shd.shd_inp import load_shd_inp_nml
from er3t.rtm.shd.shdom import shdom_ng


def _field(data):
    return {"data": np.asarray(data)}


def _atmosphere():
    return SimpleNamespace(
        lev={
            "altitude": _field([0.0, 1.0, 2.0]),
            "temperature": _field([300.0, 290.0, 280.0]),
        },
        lay={
            "altitude": _field([0.5, 1.5]),
            "thickness": _field([1.0, 1.0]),
        },
    )


def _cloud():
    return SimpleNamespace(
        lev={"altitude": _field([0.5, 1.5])},
        lay={"extinction": _field(np.array([[[2.0]]]))},
    )


def _shdom_namelist_wrapper(aeria_solver):
    return SimpleNamespace(
        Ng_=1,
        nml=[{}],
        aeria_solver=aeria_solver,
        fname_prp="column.prp",
        fname_sfc="NONE",
        fname_ckd="column.ckd",
        overwrite=True,
        force=False,
        fnames_sav=["state.sav"],
        Nx=1,
        Ny=1,
        Nz=3,
        Nmu=16,
        Nphi=32,
        target="flux",
        solver="IPA",
        wvl=556.0,
    )


def test_shdom_input_defaults_to_shdom_solver():
    defaults = load_shd_inp_nml()["shdom_nml_init"]
    assert defaults["SOLVER"] == "SHDOM"
    assert defaults["PROPERTY_LAYER_ADAPTER"] == "CONSERVATIVE"


def test_shdom_ng_emits_selected_aeria_solver():
    wrapper = _shdom_namelist_wrapper("DISORT")
    shdom_ng.nml_init(wrapper)

    assert wrapper.nml[0]["SOLVER"] == "DISORT"
    assert wrapper.nml[0]["INSAVEFILE"] == "NONE"
    assert wrapper.nml[0]["OUTSAVEFILE"] == "NONE"


def test_shdom_ng_normalizes_and_validates_aeria_solver():
    assert shdom_ng._normalize_aeria_solver(" disort ") == "DISORT"

    for invalid in (None, "mystic"):
        try:
            shdom_ng._normalize_aeria_solver(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Expected {invalid!r} to be rejected")


def test_merged_grid_preserves_layer_ownership():
    atm = _atmosphere()
    cloud = _cloud()

    levels = merged_level_grid(atm, cloud, alt_toa=3.0)
    np.testing.assert_allclose(levels, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    midpoints = 0.5 * (levels[:-1] + levels[1:])
    extinction, valid = remap_cloud_layers(cloud, midpoints, "extinction")
    np.testing.assert_allclose(extinction[0, 0], [0.0, 2.0, 2.0, 0.0, 0.0])
    np.testing.assert_array_equal(valid, [False, True, True, False, False])
    np.testing.assert_allclose(
        temperatures_on_grid(atm, levels), [300.0, 295.0, 290.0, 285.0, 280.0, 280.0]
    )


def test_ckd_coefficients_are_located_at_physical_layer_midpoints(tmp_path):
    atmosphere = _atmosphere()
    absorption = SimpleNamespace(
        wvl_max_=500.0,
        wvl_min_=499.0,
        coef={
            "weight": _field([1.0]),
            "solar": _field([1.0]),
            "abso_coef": _field([[0.01], [0.02]]),
        },
    )
    wrapper = object.__new__(shd_atm_1d)
    wrapper.quiet = True
    wrapper.atm_sca = np.array([0.1, 0.2])
    wrapper.nml = {"NZ": {"data": 4}}

    output = tmp_path / "profile.ckd"
    wrapper.gen_shd_ckd_file(output, atmosphere, absorption, alt_toa=3.0)
    lines = output.read_text().splitlines()
    altitude = [float(line.split()[0]) for line in lines[7:11]]

    np.testing.assert_allclose(altitude, [3.0, 1.5, 0.5, 0.0])
    assert wrapper.nml["NZ"]["data"] == 4


def test_ascii_propgen_output_becomes_layer_optics_and_level_temperature(tmp_path):
    output = tmp_path / "small.prp"
    output.write_text(
        "Tabulated phase function property file\n"
        "1 1 2\n"
        "1.0 1.0 0.5 2.0\n"
        "1\n"
        "2 0.0 0.5\n"
        "1 1 1 295.0 0.2 0.9 1\n"
        "1 1 2 275.0 0.4 0.8 1\n"
    )

    convert_propgen_to_layer_level(output, [0.0, 1.0, 3.0], [300.0, 280.0, 250.0])
    lines = output.read_text().splitlines()
    records = [line.split() for line in lines[-3:]]

    assert [int(value) for value in lines[1].split()] == [1, 1, 3]
    np.testing.assert_allclose([float(value) for value in lines[2].split()[2:]], [0, 1, 3])
    np.testing.assert_allclose([float(row[3]) for row in records], [300, 280, 250])
    np.testing.assert_allclose([float(row[4]) for row in records], [0.2, 0.4, 0.0])


def test_binary_propgen_output_is_converted_to_supported_ascii(tmp_path):
    output = tmp_path / "large.prp"
    output.write_text(
        "Tabulated phase function property file\n"
        "2 1 2\n"
        "1.0 1.0 0.5 2.0\n"
        "1\n"
        "2 0.0 0.5\n"
        "! The following provides information for interpreting binary data:\n"
        "! .sHdOmNG-prp\n"
        "! 2,1,2,4\n"
    )
    payload = tmp_path / "large.prp.sHdOmNG-prp"
    with payload.open("wb") as stream:
        stream.write(np.array([295, 275, 295, 275], dtype=np.float32).tobytes())
        stream.write(np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32).tobytes())
        stream.write(np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32).tobytes())
        stream.write(np.array([1, 1, 1, 1], dtype=np.int16).tobytes())

    convert_propgen_to_layer_level(output, [0.0, 1.0, 3.0], [300.0, 280.0, 250.0])
    lines = output.read_text().splitlines()
    records = [line.split() for line in lines[-6:]]

    assert not payload.exists()
    assert not any("binary data" in line for line in lines)
    np.testing.assert_allclose(
        [float(row[4]) for row in records], [0.2, 0.4, 0.0, 0.6, 0.8, 0.0]
    )
