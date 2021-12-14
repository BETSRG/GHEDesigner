# Jack C. Cook
# Friday, December 10, 2021

import ghedt as dt


class GeometricConstraints:
    def __init__(self, length: float = None, width: float = None,
                 B_min: float = None, B_max_x: float = None,
                 B_max_y: float = None, outer_constraints: list = None,
                 no_go: list = None, unconstrained: bool = False):
        # Spacing parameters in meters
        self.B_max_x = B_max_x
        self.B_max_y = B_max_y
        self.B_min = B_min

        # The outer constraint parameter handling
        self.width = width
        self.length = length
        # If both length and width are none, then do the outer constraints
        if length != None and width != None:
            self.outer_constraints = \
                dt.utilities.make_rectangle_perimeter(self.length, self.width)
        # Make sure the user understands the logic
        elif length == None and width != None or length != None and width == None:
            raise ValueError('The `length` and `width` parameters both need to '
                             'be defined if one is.')
        # If the outer constraints are not given and the problem is
        # constrained, then throw an error.
        elif length == None and width == None:
            if outer_constraints == None and unconstrained == False:
                raise ValueError('Either `length` and `width` need to be '
                                 'provided, or a list of `outer_constraints`,'
                                 'or `unconstrained` needs to be True.')
            else:
                self.outer_constraints = outer_constraints
        # Handling for a list or a list of lists to occur later
        self.no_go = no_go
