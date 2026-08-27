"""Vertical-grid utilities for the aeria3d interface.

The public atmosphere objects store optical properties in layers and
temperatures at both layers and levels.  aeria3d's physical ``LAYER`` mode
instead expects a boundary grid with one fewer optical-property records than
temperature records.  The helpers in this module keep that conversion in one
place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def merged_level_grid(atm, cld=None, alt_toa=None, tolerance=1.0e-6):
    """Return a strictly increasing grid of physical layer boundaries."""
    atm_z = _altitudes(atm.lev["altitude"]["data"], "atmosphere levels")
    grids = [atm_z]
    if cld is not None:
        grids.append(_altitudes(cld.lev["altitude"]["data"], "cloud levels"))

    candidates = np.sort(np.concatenate(grids).astype(float))
    merged = [candidates[0]]
    for value in candidates[1:]:
        if value - merged[-1] > tolerance:
            merged.append(value)
    z_levels = np.asarray(merged)
    if alt_toa is not None and alt_toa > z_levels[-1]:
        z_levels = np.append(z_levels, float(alt_toa))
    if z_levels[0] > 0.0:
        z_levels = np.insert(z_levels, 0, 0.0)
    if z_levels.size < 2 or np.any(np.diff(z_levels) <= 0.0):
        raise ValueError("The aeria3d altitude levels must be strictly increasing.")
    return z_levels


def temperatures_on_grid(atm, z):
    """Interpolate atmospheric level temperatures to *z*."""
    atm_z = _altitudes(atm.lev["altitude"]["data"], "atmosphere levels")
    atm_t = _one_dimensional(atm.lev["temperature"]["data"], "temperatures")
    if atm_z.size != atm_t.size:
        raise ValueError("Atmosphere level altitudes and temperatures must match.")
    return np.interp(np.asarray(z, dtype=float), atm_z, atm_t)


def remap_cloud_layers(cld, target_altitude, field, *, fill_value=0.0):
    """Sample a cloud layer field at target layer midpoints.

    Values remain piecewise constant inside their source cells.  This preserves
    layer ownership when atmospheric boundaries subdivide a cloud layer.
    """
    source_z = _altitudes(cld.lev["altitude"]["data"], "cloud levels")
    values = np.asarray(cld.lay[field]["data"])
    if values.ndim != 3 or values.shape[2] != source_z.size - 1:
        raise ValueError(
            f"Cloud field {field!r} must have shape (NX, NY, number of layers)."
        )

    target = np.asarray(target_altitude, dtype=float)
    source_index = np.searchsorted(source_z, target, side="right") - 1
    valid = (source_index >= 0) & (source_index < source_z.size - 1)
    result = np.full((*values.shape[:2], target.size), fill_value, dtype=values.dtype)
    result[:, :, valid] = values[:, :, source_index[valid]]
    return result, valid


def convert_propgen_to_layer_level(fname, z_levels, temperature_levels):
    """Convert propgen midpoint output to aeria3d LAYER/LEVEL records.

    ``propgen`` is intentionally retained for particle/Rayleigh phase mixing.
    Its vertical points are supplied at physical layer midpoints.  This routine
    changes only ownership metadata and temperature records: midpoint optical
    values become cell-owned, while temperatures are written at all boundaries.
    """
    path = Path(fname)
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        raise ValueError(f"Incomplete propgen property file: {fname}")

    nx, ny, nz_layers = (int(value) for value in lines[1].split()[:3])
    z_levels = _altitudes(z_levels, "aeria3d levels")
    temperature_levels = _one_dimensional(
        temperature_levels, "aeria3d level temperatures"
    )
    if z_levels.size != nz_layers + 1:
        raise ValueError(
            "propgen must provide one midpoint optical record per aeria3d layer."
        )
    if temperature_levels.size != z_levels.size:
        raise ValueError("One temperature is required at every aeria3d level.")

    grid_tokens = lines[2].split()
    if len(grid_tokens) < nz_layers + 2:
        raise ValueError(f"Invalid propgen grid header: {fname}")
    dx, dy = (float(value) for value in grid_tokens[:2])
    record_count = nx * ny * nz_layers
    binary_marker = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("! The following provides information")
        ),
        None,
    )

    binary_path = None
    if binary_marker is None:
        if len(lines) < 4 + record_count:
            raise ValueError(f"Missing propgen property records: {fname}")
        phase_lines = lines[4:-record_count]
        records = _read_ascii_records(lines[-record_count:], record_count)
    else:
        phase_lines = lines[4:binary_marker]
        if binary_marker + 1 >= len(lines):
            raise ValueError(f"Missing propgen binary payload name: {fname}")
        postfix = lines[binary_marker + 1].lstrip("! ").strip()
        binary_path = Path(f"{path}{postfix}")
        records = _read_binary_records(binary_path, record_count)

    temporary = path.with_name(f".{path.name}.layer-level.tmp")
    try:
        with temporary.open("w") as output:
            output.write(f"{lines[0]}\n")
            output.write(f"{nx:6d} {ny:6d} {z_levels.size:6d}\n")
            z_text = " ".join(f"{value:.8f}" for value in z_levels)
            output.write(f"{dx:.8f} {dy:.8f} {z_text}\n")
            output.write(f"{lines[3]}\n")
            for line in phase_lines:
                output.write(f"{line}\n")

            for ix in range(nx):
                for iy in range(ny):
                    start = (ix * ny + iy) * nz_layers
                    for iz in range(nz_layers):
                        extinction, albedo, phase = records(start + iz)
                        output.write(
                            f"{ix + 1:5d} {iy + 1:5d} {iz + 1:5d} "
                            f"{temperature_levels[iz]:15.8e} "
                            f"{extinction:15.8e} {albedo:15.8e} {phase:5d}\n"
                        )
                    output.write(
                        f"{ix + 1:5d} {iy + 1:5d} {z_levels.size:5d} "
                        f"{temperature_levels[-1]:15.8e} "
                        f"{0.0:15.8e} {0.0:15.8e} {1:5d}\n"
                    )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    if binary_path is not None and binary_path.exists():
        binary_path.unlink()


def _read_ascii_records(lines, expected):
    records = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 7:
            raise ValueError("Invalid ASCII propgen property record.")
        records.append((float(tokens[4]), float(tokens[5]), int(tokens[6])))
    if len(records) != expected:
        raise ValueError("Unexpected number of ASCII propgen property records.")
    return records.__getitem__


def _read_binary_records(path, count):
    bytes_per_record = (
        3 * np.dtype(np.float32).itemsize + np.dtype(np.int16).itemsize
    )
    expected_size = count * bytes_per_record
    if not path.exists() or path.stat().st_size != expected_size:
        raise ValueError(f"Invalid propgen binary property payload: {path}")
    extinction = np.memmap(
        path, dtype=np.float32, mode="r", offset=count * 4, shape=count
    )
    albedo = np.memmap(
        path, dtype=np.float32, mode="r", offset=count * 8, shape=count
    )
    phase = np.memmap(
        path, dtype=np.int16, mode="r", offset=count * 12, shape=count
    )

    def get_record(index):
        return float(extinction[index]), float(albedo[index]), int(phase[index])

    return get_record


def _one_dimensional(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name.capitalize()} must be a finite one-dimensional array.")
    return result


def _altitudes(values, name):
    result = _one_dimensional(values, name)
    if result.size < 2 or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name.capitalize()} must be strictly increasing.")
    return result
