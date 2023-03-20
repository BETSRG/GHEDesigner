from abc import abstractmethod

from ghedesigner.constants import RAD_TO_DEG
from ghedesigner.enums import DesignGeomType


class GeometricConstraints:
    def __init__(self):
        self.type = None

    @abstractmethod
    def to_input(self):
        pass


class GeometricConstraintsNearSquare(GeometricConstraints):
    """
    Geometric constrains for near square design algorithm
    """

    def __init__(self, b: float, length: float):
        super().__init__()
        self.b = b
        self.length = length
        self.type = DesignGeomType.NEARSQUARE

    def to_input(self) -> dict:
        return {'length': self.length,
                'b': self.b,
                'method': DesignGeomType.NEARSQUARE.name}


class GeometricConstraintsRectangle(GeometricConstraints):
    """
    Geometric constraints for rectangular design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float):
        super().__init__()
        self.width = width
        self.length = length
        self.b_min = b_min
        self.b_max_x = b_max_x
        self.type = DesignGeomType.RECTANGLE

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.b_min,
                'b_max': self.b_max_x,
                'method': DesignGeomType.RECTANGLE.name}


class GeometricConstraintsBiRectangle(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float, b_max_y: float):
        super().__init__()
        self.width = width
        self.length = length
        self.b_min = b_min
        self.b_max_x = b_max_x
        self.b_max_y = b_max_y
        self.type = DesignGeomType.BIRECTANGLE

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.b_min,
                'b_max_x': self.b_max_x,
                'b_max_y': self.b_max_y,
                'method': DesignGeomType.BIRECTANGLE.name}


class GeometricConstraintsBiRectangleConstrained(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle constrained design algorithm
    """

    def __init__(self, b_min: float, b_max_x: float, b_max_y: float, property_boundary, no_go_boundaries):
        super().__init__()
        self.b_min = b_min
        self.b_max_x = b_max_x
        self.b_max_y = b_max_y
        self.property_boundary = property_boundary
        self.no_go_boundaries = no_go_boundaries
        self.type = DesignGeomType.BIRECTANGLECONSTRAINED

    def to_input(self) -> dict:
        return {'b_min': self.b_min,
                'b_max_x': self.b_max_x,
                'b_max_y': self.b_max_y,
                'property_boundary': self.property_boundary,
                'no_go_boundaries': self.no_go_boundaries,
                'method': DesignGeomType.BIRECTANGLECONSTRAINED.name}


class GeometricConstraintsBiZoned(GeometricConstraintsBiRectangle):
    """
    Geometric constraints for bi-zoned design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float, b_max_y: float):
        super().__init__(width, length, b_min, b_max_x, b_max_y)
        self.type = DesignGeomType.BIZONEDRECTANGLE

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.b_min,
                'b_max_x': self.b_max_x,
                'b_max_y': self.b_max_y,
                'method': DesignGeomType.BIZONEDRECTANGLE.name}


class GeometricConstraintsRowWise(GeometricConstraints):
    """
    Geometric constraints for rowwise design algorithm
    """

    def __init__(self,
                 perimeter_spacing_ratio: float,
                 min_spacing: float,
                 max_spacing: float,
                 spacing_step: float,
                 min_rotation: float,
                 max_rotation: float,
                 rotate_step: float,
                 property_boundary,
                 no_go_boundaries):
        super().__init__()
        self.perimeter_spacing_ratio = perimeter_spacing_ratio
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.spacing_step = spacing_step
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.rotate_step = rotate_step
        self.property_boundary = property_boundary
        self.no_go_boundaries = no_go_boundaries
        self.type = DesignGeomType.ROWWISE

    def to_input(self) -> dict:
        return {'perimeter_spacing_ratio': self.perimeter_spacing_ratio,
                'min_spacing': self.min_spacing,
                'max_spacing': self.max_spacing,
                'spacing_step': self.spacing_step,
                'min_rotation': self.min_rotation * RAD_TO_DEG,
                'max_rotation': self.max_rotation * RAD_TO_DEG,
                'rotate_step': self.rotate_step,
                'property_boundary': self.property_boundary,
                'no_go_boundaries': self.no_go_boundaries,
                'method': DesignGeomType.ROWWISE.name}
