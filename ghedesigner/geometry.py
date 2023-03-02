class GeometricConstraints:
    pass  # TODO: Pull in common functionality!


class GeometricConstraintsNearSquare(GeometricConstraints):
    """
    Geometric constrains for near square design algorithm
    """

    def __init__(self, b: float, length: float):
        super().__init__()
        self.B = b
        self.length = length

    def to_input(self) -> dict:
        return {'length': self.length,
                'b': self.B,
                'method': 'nearsquare'}


class GeometricConstraintsRectangle(GeometricConstraints):
    """
    Geometric constraints for rectangular design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float):
        super().__init__()
        self.width = width
        self.length = length
        self.B_min = b_min
        self.B_max_x = b_max_x

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.B_min,
                'b_max': self.B_max_x,
                'method': 'rectangle'}


class GeometricConstraintsBiRectangle(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle design algorithm
    """
    def __init__(self, width: float, length: float, b_min: float, b_max_x: float, b_max_y: float):
        super().__init__()
        self.width = width
        self.length = length
        self.B_min = b_min
        self.B_max_x = b_max_x
        self.B_max_y = b_max_y

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.B_min,
                'b_max_x': self.B_max_x,
                'b_max_y': self.B_max_y,
                'method': 'birectangle'}


class GeometricConstraintsBiRectangleConstrained(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle constrained design algorithm
    """

    def __init__(self, b_min: float, b_max_y: float, b_max_x: float):
        super().__init__()
        self.B_min = b_min
        self.B_max_y = b_max_y
        self.B_max_x = b_max_x


class GeometricConstraintsBiZoned(GeometricConstraintsBiRectangle):
    """
    Geometric constraints for bi-zoned design algorithm
    """

    def __init__(self, width: float, length: float, b_min: float, b_max_x: float, b_max_y: float):
        super().__init__(width, length, b_min, b_max_x, b_max_y)

    def to_input(self) -> dict:
        return {'length': self.length,
                'width': self.width,
                'b_min': self.B_min,
                'b_max_x': self.B_max_x,
                'b_max_y': self.B_max_y,
                'method': 'bizonedrectangle'}


class GeometricConstraintsRowWise(GeometricConstraints):
    """
    Geometric constraints for rowwise design algorithm
    """

    def __init__(self, p_spacing: float,
                 spacing_start: float,
                 spacing_stop: float,
                 spacing_step: float,
                 rotate_step: float,
                 rotate_stop: float,
                 rotate_start: float,
                 property_boundary,
                 ng_zones):
        super().__init__()
        self.pSpac = p_spacing
        self.spacStart = spacing_start
        self.spacStop = spacing_stop
        self.spacStep = spacing_step
        self.rotateStep = rotate_step
        self.rotateStop = rotate_stop
        self.rotateStart = rotate_start
        self.propBound = property_boundary
        self.ngZones = ng_zones
