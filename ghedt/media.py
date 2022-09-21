

class GeometricConstraints:
    def __init__(self, length: float = None, width: float = None,
                 B: float = None, B_min: float = None,
                 B_max_x: float = None, B_max_y: float = None,
                 outer_constraints: list = None, no_go: list = None, pSpac: float = None, spacStart: float = None
                 , spacStop: float = None, spacStep: float = None, rotateStep: float = None, propBound: list = None,
                 ngZones: list = None, rotateStart: float = None, rotateStop: float = None, Directory="",
                 pdfOutputName="Graphs.pdf"):
        # Spacing parameters in meters
        self.B = B
        self.B_max_x = B_max_x
        self.B_max_y = B_max_y
        self.B_min = B_min
        # Length and width constraints
        self.length = length
        self.width = width
        # Outer constraints described as a polygon
        self.outer_constraints = outer_constraints
        # TODO: Handling for a list or a list of lists to occur later
        # Note: the entirety of the no-go zone should fall inside the
        # outer_constraints
        self.no_go = no_go
        self.pSpac = pSpac
        self.spacStart = spacStart
        self.spacStop = spacStop
        self.spacStep = spacStep
        self.rotateStep = rotateStep
        self.propBound = propBound
        self.ngZones = ngZones
        self.rotateStart = rotateStart
        self.rotateStop = rotateStop
        self.Directory = Directory
        self.pdfOutputName = pdfOutputName

    def check_inputs(self, method):
        # The required instances for the near-square design is self.B
        if method == 'near-square':
            assert self.B is not None
            assert self.length is not None
        elif method == 'rectangle':
            assert self.width is not None
            assert self.length is not None
            assert self.B_min is not None
            assert self.B_max_x is not None
        elif method == 'bi-rectangle' or method == 'bi-zoned':
            assert self.width is not None
            assert self.length is not None
            assert self.B_min is not None
            assert self.B_max_x is not None
            assert self.B_max_y is not None
        elif method == 'row-wise':
            assert self.pSpac is not None
            assert self.spacStart is not None
            assert self.spacStop is not None
            assert self.spacStep is not None
            assert self.rotateStep is not None
            assert self.rotateStop is not None
            assert self.rotateStart is not None
            assert self.propBound is not None
            assert self.ngZones is not None

        return
