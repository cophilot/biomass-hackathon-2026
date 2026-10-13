from dataclasses import dataclass
from typing import TypeVar, Generic


T = TypeVar('T')

@dataclass
class DataLayer(Generic[T]):
    data: T
    timestamp: str

S = TypeVar('S')

@dataclass
class DataLayer3D(DataLayer[list[list[list[S]]]]):
    "Data Layer of 3D data"
    
    def get(self, x: int, y: int, z: int) -> S:
        return self.data[z][y][x]
    

@dataclass
class FloatDataLayer3D(DataLayer3D[float]):
    "Float Data Layer of 3D data"
    
@dataclass
class DataCube:
    als: FloatDataLayer3D
    tls: FloatDataLayer3D
    p_band: FloatDataLayer3D
    gedi: FloatDataLayer3D
    x_dim: int
    y_dim: int
