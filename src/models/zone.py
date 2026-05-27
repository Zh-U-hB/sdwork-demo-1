from pydantic import BaseModel, Field


class Point3D(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Dimensions(BaseModel):
    length: float = Field(gt=0, description="X-axis dimension in meters")
    width: float = Field(gt=0, description="Y-axis dimension in meters")
    height: float = Field(gt=0, description="Z-axis dimension in meters")


class Zone(BaseModel):
    name: str = Field(description="Unique zone name, e.g. 'Living Room', 'Bedroom'")
    origin: Point3D = Field(default_factory=Point3D, description="Origin point (x, y, z) in meters")
    dimensions: Dimensions = Field(description="Dimensions (length, width, height) in meters")


class BuildingModel(BaseModel):
    building_name: str = Field(default="Unnamed Building")
    zones: list[Zone] = Field(default_factory=list)
