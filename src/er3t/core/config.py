"""Explicit runtime settings replacing implicit global configuration over time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .resources import cache_dir, resource_path


@dataclass(slots=True)
class Settings:
    """Configuration shared by preprocessing and RTM workflows.

    Workflows should accept a ``Settings`` instance instead of reading mutable
    process-wide state.
    """

    wavelength: float = 650.0
    date: datetime = field(default_factory=datetime.now)
    solar_zenith_angle: float = 0.0
    solar_azimuth_angle: float = 0.0
    sensor_zenith_angle: float = 0.0
    sensor_azimuth_angle: float = 0.0
    sensor_altitude: float = 705.0
    target: str = "3d radiance"
    solver: str = "mcarats"
    atmospheric_profile: Path = field(
        default_factory=lambda: resource_path("atmmod", "afglus.dat")
    )
    absorption: str = "abs_16g"
    surface_albedo: float = 0.03
    phase_cloud: str = "mie"
    Nphoton: float = 1e8
    Ncpu: int = 12
    fdir_tmp: Path = field(default_factory=cache_dir)
    output_tag: str = "rtm-out_rad-3d"
    overwrite: bool = True
    verbose: bool = True
    earth_radius: float = 6371.009


def default_settings() -> Settings:
    """Create an independent settings object with project defaults."""

    return Settings()
