from pygfunction.boreholes import Borehole


class GHEBorehole(Borehole):
    def __init__(self, height, buried_depth, radius, x, y, tilt=0, orientation=0):
        super().__init__(height, buried_depth, radius, x, y, tilt, orientation)

    def to_input(self):
        return {'buried_depth': self.D, 'diameter': self.r_b * 2.0}
