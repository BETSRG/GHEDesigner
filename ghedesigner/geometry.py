class GeometricConstraints:
    pass  # TODO: Pull in common functionality!


class GeometricConstraintsNearSquare(GeometricConstraints):
    """
    Geometric constrains for near square design algorithm
    """

    def __init__(self, b: float, length: float):
        super().__init__()
        self.b = b
        self.length = length

    def to_input(self) -> dict:
        return {'length': self.length,
                'b': self.b,
                'method': 'nearsquare'}


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

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.b_min,
                'b_max': self.b_max_x,
                'method': 'rectangle'}


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

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.b_min,
                'b_max_x': self.b_max_x,
                'b_max_y': self.b_max_y,
                'method': 'birectangle'}


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

    def to_input(self) -> dict:
        return {'b_min': self.b_min,
                'b_max_x': self.b_max_x,
                'b_max_y': self.b_max_y,
                'property_boundary': self.property_boundary,
                'no_go_boundaries': self.no_go_boundaries,
                'method': 'birectangleconstrained'}


class GeometricConstraintsBiZoned(GeometricConstraintsBiRectangle):
    """
    Geometric constraints for bi-zoned design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float, b_max_y: float):
        super().__init__(width, length, b_min, b_max_x, b_max_y)

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.b_min,
                'b_max_x': self.b_max_x,
                'b_max_y': self.b_max_y,
                'method': 'bizonedrectangle'}


class GeometricConstraintsRowWise(GeometricConstraints):
    """
    Geometric constraints for rowwise design algorithm
    """

    def __init__(self,
                 perimeter_spacing_ratio: float,
                 spacing_start: float,
                 spacing_stop: float,
                 spacing_step: float,
                 rotate_start: float,
                 rotate_stop: float,
                 rotate_step: float,
                 property_boundary,
                 no_go_boundaries):
        super().__init__()
        self.perimeter_spacing_ratio = perimeter_spacing_ratio
        self.spacing_start = spacing_start
        self.spacing_stop = spacing_stop
        self.spacing_step = spacing_step
        self.rotate_start = rotate_start
        self.rotate_stop = rotate_stop
        self.rotate_step = rotate_step
        self.property_boundary = property_boundary
        self.no_go_boundaries = no_go_boundaries

    def to_input(self) -> dict:
        return {'perimeter_spacing_ratio': self.perimeter_spacing_ratio,
                'spacing_start': self.spacing_start,
                'spacing_stop': self.spacing_stop,
                'spacing_step': self.spacing_step,
                'rotate_start': self.rotate_start,
                'rotate_stop': self.rotate_stop,
                'rotate_step': self.rotate_step,
                'property_boundary': self.property_boundary,
                'no_go_boundaries': self.no_go_boundaries,
                'method': 'rowwise'}
