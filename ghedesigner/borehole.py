from pygfunction.boreholes import Borehole


class GHEBorehole(Borehole):
    def __init__(self, H, D, r_b, x, y, tilt=0, orientation=0):
        super().__init__(H, D, r_b, x, y, tilt, orientation)

    def to_input(self):
        return {'buried_depth': self.D, 'radius': self.r_b}
