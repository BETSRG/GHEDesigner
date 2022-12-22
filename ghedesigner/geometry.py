class GeometricConstraints:
    def __init__(
            self,
            length: float = None,
            width: float = None,
            b: float = None,
            b_min: float = None,
            b_max_x: float = None,
            b_max_y: float = None,
            outer_constraints: list = None,
            no_go: list = None,
            p_spacing: float = None,
            spacing_start: float = None,
            spacing_stop: float = None,
            spacing_step: float = None,
            rotate_step: float = None,
            prop_bound: list = None,
            ng_zones: list = None,
            rotate_start: float = None,
            rotate_stop: float = None
    ):
        # Spacing parameters in meters
        self.B = b
        self.B_max_x = b_max_x
        self.B_max_y = b_max_y
        self.B_min = b_min
        # Length and width constraints
        self.length = length
        self.width = width
        # Outer constraints described as a polygon
        self.outer_constraints = outer_constraints
        # TODO: Handling for a list or a list of lists to occur later
        # Note: the entirety of the no-go zone should fall inside the
        # outer_constraints
        self.no_go = no_go
        self.pSpac = p_spacing
        self.spacStart = spacing_start
        self.spacStop = spacing_stop
        self.spacStep = spacing_step
        self.rotateStep = rotate_step
        self.propBound = prop_bound
        self.ngZones = ng_zones
        self.rotateStart = rotate_start
        self.rotateStop = rotate_stop

    def check_inputs(self, method):
        # The required instances for the near-square design is self.B
        if method == "near-square":
            assert self.B is not None
            assert self.length is not None
        elif method == "rectangle":
            assert self.width is not None
            assert self.length is not None
            assert self.B_min is not None
            assert self.B_max_x is not None
        elif method == "bi-rectangle" or method == "bi-zoned":
            assert self.width is not None
            assert self.length is not None
            assert self.B_min is not None
            assert self.B_max_x is not None
            assert self.B_max_y is not None
        elif method == "row-wise":
            assert self.pSpac is not None
            assert self.spacStart is not None
            assert self.spacStop is not None
            assert self.spacStep is not None
            assert self.rotateStep is not None
            assert self.rotateStop is not None
            assert self.rotateStart is not None
            assert self.propBound is not None
            assert self.ngZones is not None
