"""Default EnergyPlus simulation parameters.

All parameters are gathered in :class:`ConverterDefaults` and can be
overridden piecemeal by the caller before passing to
:func:`scripts.idf_converter.convert_and_run`.

Example — override only the location while keeping everything else at
defaults::

    defaults = make_default_settings()
    defaults.location.latitude  = 39.93
    defaults.location.longitude = 116.28
    defaults.location.time_zone = 8.0
    defaults.location.elevation = 44.0
    defaults.location.name      = "Beijing"
    result_dir = convert_and_run(model_dict, output_dir="output/sim", defaults=defaults)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Individual parameter types
# ---------------------------------------------------------------------------


@dataclass
class MaterialDef:
    """A single opaque material (Material IDD object)."""
    name: str
    roughness: str
    thickness: float          # m
    conductivity: float       # W/(m·K)
    density: float            # kg/m³
    specific_heat: float      # J/(kg·K)


@dataclass
class GlazingDef:
    """Simple glazing system (WindowMaterial:SimpleGlazingSystem)."""
    name: str
    u_factor: float                        # W/(m²·K)
    solar_heat_gain_coefficient: float     # dimensionless
    visible_transmittance: float = 0.6     # dimensionless


@dataclass
class ConstructionDef:
    """A construction assembly referencing material names in order (outside→inside)."""
    name: str
    layers: list[str]


@dataclass
class ScheduleTypeDef:
    """ScheduleTypeLimits object."""
    name: str
    lower_limit: float
    upper_limit: float
    numeric_type: str   # "Continuous" | "Discrete"
    unit_type: str      # "Dimensionless" | "Temperature" | "ActivityLevel" …


@dataclass
class ScheduleDef:
    """Schedule:Compact object.

    ``data`` is a flat list of EnergyPlus compact-schedule strings, e.g.::

        ["Through: 12/31",
         "For: Weekdays",
         "Until: 08:00,0.0",
         "Until: 18:00,1.0",
         "Until: 24:00,0.0",
         "For: AllOtherDays",
         "Until: 24:00,0.0"]
    """
    name: str
    type_limits_name: str
    data: list[str]


@dataclass
class LocationDef:
    """Site:Location object."""
    name: str
    latitude: float     # decimal degrees, N positive
    longitude: float    # decimal degrees, E positive
    time_zone: float    # UTC offset, hours
    elevation: float    # m above sea level


@dataclass
class PeopleDef:
    """People object parameters (density-based)."""
    people_per_floor_area: float = 0.05   # persons/m²
    number_of_people_schedule_name: str = "OccupancySchedule"
    activity_level_schedule_name: str = "ActivityLevelSchedule"
    fraction_radiant: float = 0.3


@dataclass
class LightDef:
    """Lights object parameters (density-based)."""
    watts_per_floor_area: float = 10.0    # W/m²
    schedule_name: str = "AlwaysOnSchedule"
    return_air_fraction: float = 0.0
    fraction_radiant: float = 0.32
    fraction_visible: float = 0.25


@dataclass
class EquipmentDef:
    """Electric equipment object parameters (density-based)."""
    watts_per_floor_area: float = 8.0     # W/m²
    schedule_name: str = "AlwaysOnSchedule"
    fraction_latent: float = 0.0
    fraction_radiant: float = 0.30
    fraction_lost: float = 0.0


@dataclass
class HvacDef:
    """HVAC thermostat + IdealLoadsAirSystem parameters."""
    thermostat_name: str = "DefaultThermostat"
    heating_setpoint_schedule_name: str = "HeatingSetpointSchedule"
    cooling_setpoint_schedule_name: str = "CoolingSetpointSchedule"


@dataclass
class WindowDef:
    """Window-to-wall ratio (fallback) and glazing construction for fenestration.

    When the BuildingModel JSON includes per-zone ``windows`` vertices, the
    converter uses those surfaces with ``construction_name`` (glass layers).
    ``wwr`` applies only if no geometry windows are present.
    """
    wwr: float = 0.0                       # 0.0 = no windows; 0.4 = 40%
    construction_name: str = "WindowConstruction"


# ---------------------------------------------------------------------------
# Master defaults container
# ---------------------------------------------------------------------------


@dataclass
class ConverterDefaults:
    """All default parameters used by :mod:`scripts.idf_converter`.

    Replace any sub-dataclass to customise only part of the model while
    keeping remaining fields at their default values.
    """
    location: LocationDef = field(default_factory=LocationDef)
    opaque_materials: list[MaterialDef] = field(default_factory=list)
    glazing_materials: list[GlazingDef] = field(default_factory=list)
    constructions: list[ConstructionDef] = field(default_factory=list)
    schedule_types: list[ScheduleTypeDef] = field(default_factory=list)
    schedules: list[ScheduleDef] = field(default_factory=list)
    people: PeopleDef = field(default_factory=PeopleDef)
    lights: LightDef = field(default_factory=LightDef)
    equipment: EquipmentDef = field(default_factory=EquipmentDef)
    hvac: HvacDef = field(default_factory=HvacDef)
    window: WindowDef = field(default_factory=WindowDef)


# ---------------------------------------------------------------------------
# Factory — builds the default ConverterDefaults instance
# ---------------------------------------------------------------------------


def make_default_settings() -> ConverterDefaults:
    """Return a fully populated :class:`ConverterDefaults` with sensible defaults.

    The returned object is a fresh instance every call; mutate freely without
    affecting later calls.
    """

    # ── Location ─────────────────────────────────────────────────────────────
    location = LocationDef(
        name="Shenzhen",
        latitude=22.55,
        longitude=114.10,
        time_zone=8.0,
        elevation=5.0,
    )

    # ── Materials ─────────────────────────────────────────────────────────────
    opaque_materials = [
        MaterialDef(
            name="Concrete200",
            roughness="MediumRough",
            thickness=0.20,
            conductivity=1.63,
            density=2240.0,
            specific_heat=900.0,
        ),
        MaterialDef(
            name="Insulation50",
            roughness="MediumSmooth",
            thickness=0.05,
            conductivity=0.04,
            density=30.0,
            specific_heat=840.0,
        ),
        MaterialDef(
            name="Gypsum13",
            roughness="Smooth",
            thickness=0.013,
            conductivity=0.16,
            density=784.0,
            specific_heat=830.0,
        ),
    ]

    glazing_materials = [
        GlazingDef(
            name="SimpleGlazing",
            u_factor=2.7,
            solar_heat_gain_coefficient=0.25,
            visible_transmittance=0.6,
        )
    ]

    # ── Constructions ─────────────────────────────────────────────────────────
    constructions = [
        ConstructionDef(
            name="ExteriorWallConstruction",
            layers=["Concrete200", "Insulation50", "Gypsum13"],
        ),
        ConstructionDef(
            name="InteriorWallConstruction",
            layers=["Gypsum13", "Gypsum13"],
        ),
        ConstructionDef(
            name="ExteriorRoofConstruction",
            layers=["Concrete200", "Insulation50", "Gypsum13"],
        ),
        ConstructionDef(
            name="FloorConstruction",
            layers=["Concrete200"],
        ),
        ConstructionDef(
            name="InteriorCeilingConstruction",
            layers=["Concrete200"],
        ),
        ConstructionDef(
            name="WindowConstruction",
            layers=["SimpleGlazing"],
        ),
    ]

    # ── Schedule type limits ──────────────────────────────────────────────────
    schedule_types = [
        ScheduleTypeDef(
            name="FractionLimits",
            lower_limit=0.0,
            upper_limit=1.0,
            numeric_type="Continuous",
            unit_type="Dimensionless",
        ),
        ScheduleTypeDef(
            name="TemperatureLimits",
            lower_limit=-100.0,
            upper_limit=200.0,
            numeric_type="Continuous",
            unit_type="Temperature",
        ),
        ScheduleTypeDef(
            name="ActivityLevelLimits",
            lower_limit=0.0,
            upper_limit=10000.0,
            numeric_type="Continuous",
            unit_type="ActivityLevel",
        ),
    ]

    # ── Schedules ─────────────────────────────────────────────────────────────
    schedules = [
        ScheduleDef(
            name="AlwaysOnSchedule",
            type_limits_name="FractionLimits",
            data=[
                "Through: 12/31",
                "For: AllDays",
                "Until: 24:00,1.0",
            ],
        ),
        ScheduleDef(
            name="OccupancySchedule",
            type_limits_name="FractionLimits",
            data=[
                "Through: 12/31",
                "For: Weekdays",
                "Until: 08:00,0.0",
                "Until: 18:00,1.0",
                "Until: 24:00,0.0",
                "For: AllOtherDays",
                "Until: 24:00,0.0",
            ],
        ),
        ScheduleDef(
            name="ActivityLevelSchedule",
            type_limits_name="ActivityLevelLimits",
            data=[
                "Through: 12/31",
                "For: AllDays",
                "Until: 24:00,120.0",
            ],
        ),
        ScheduleDef(
            name="HeatingSetpointSchedule",
            type_limits_name="TemperatureLimits",
            data=[
                "Through: 12/31",
                "For: AllDays",
                "Until: 24:00,21.0",
            ],
        ),
        ScheduleDef(
            name="CoolingSetpointSchedule",
            type_limits_name="TemperatureLimits",
            data=[
                "Through: 12/31",
                "For: AllDays",
                "Until: 24:00,26.0",
            ],
        ),
    ]

    # ── People / Lights / HVAC / Window ──────────────────────────────────────
    people = PeopleDef()
    lights = LightDef()
    hvac   = HvacDef()
    window = WindowDef()   # wwr=0.0 → no windows by default

    return ConverterDefaults(
        location=location,
        opaque_materials=opaque_materials,
        glazing_materials=glazing_materials,
        constructions=constructions,
        schedule_types=schedule_types,
        schedules=schedules,
        people=people,
        lights=lights,
        hvac=hvac,
        window=window,
        equipment=EquipmentDef(),
    )
