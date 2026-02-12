from functools import lru_cache
from math import atan, cos, inf, pi, sin, sqrt

import numpy as np

from ghedesigner.constants import DEG_TO_RAD, PI_OVER_2, RAD_TO_DEG
from ghedesigner.ghe.shape import Shapes, point_polygon_check, sort_intersections

NEIGHBORHOOD_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


class DeferredDuplicateCheckList:
    def __init__(self, spacing):
        self.points = {}
        self.buckets = {}
        self.points_to_check = []
        self.spacing = spacing
        self.partitioned = False
        self.proximity_checks_found = False
        self.largest_index = 0
        self.bucket_keys = None

    def _get_bucket_key(self, px, py):
        return int(px // self.spacing), int(py // self.spacing)

    def _get_bucket_keys(self, points_array):
        return (points_array / self.spacing).astype(int)

    def size(self):
        return len(self.points)

    def delete(self, bucket_keys_index):
        key = self.bucket_keys[bucket_keys_index]
        key = (key[0], key[1])
        point_keys = list(self.points.keys())
        point_index = point_keys[bucket_keys_index]
        index_to_delete = self.buckets[key].index(point_index)
        del self.buckets[key][index_to_delete]
        del self.points[point_index]
        del self.bucket_keys[bucket_keys_index]

    def index(self, ind):
        keys = list(self.points.keys())
        return self.points[keys[ind]]

    def partition(self):
        array_repr = self.toarray()
        bucket_keys = self._get_bucket_keys(array_repr)
        self.bucket_keys = bucket_keys.tolist()
        point_keys = list(self.points.keys())
        for i, key in enumerate(bucket_keys):
            usable_key = (key[0], key[1])
            if usable_key in self.buckets:
                self.buckets[usable_key].append(point_keys[i])
            else:
                self.buckets[usable_key] = [point_keys[i]]

        self.partitioned = True

    def find_proximity_checks(self):
        if not self.partitioned:
            self.partition()
        for i, key in enumerate(self.points):
            x_ind, y_ind = self.bucket_keys[i]
            for key_modifier in NEIGHBORHOOD_OFFSETS:
                dx, dy = key_modifier
                neighbor_key = (x_ind + dx, y_ind + dy)
                for j in self.buckets.get(neighbor_key, []):
                    if j <= key:
                        continue
                    self.points_to_check.append((key, j))
        self.proximity_checks_found = True

    def append(self, element):
        # Since the spatial partitioning is done in bulk in "partition()", we have to reset the partitioning
        # when appending a new element.
        if self.partitioned:
            self.partitioned = False
            self.buckets = {}
            self.bucket_keys = None
        # For a similar reason, the proximity checks need to be rerun when a new element is appended.
        if self.proximity_checks_found:
            self.proximity_checks_found = False
            self.points_to_check = []
        # Internally, points are contained in the self.points dictionary, but we want this classes usage
        # to be similar to a list/array. This can cause an issue when doing deletions (as we would
        # like to avoid readjusting the keys for all dict entries). In order to avoid generating an already
        # existing key, "largest_index" is used to always increment the key for the added element (regardless
        # of deletions). There is probably a better way to implement this internally, as I suspect the
        # point indexing with this implementation is slower than necessary.
        self.points[self.largest_index] = element
        self.largest_index += 1

    def extend(self, elements):
        for element in elements:
            self.append(element)

    def tolist(self):
        return list(self.points.values())

    def toarray(self):
        return np.array(self.tolist(), dtype=float)

    def find_duplicates(self, tolerance):
        if tolerance > self.spacing:
            raise ValueError("Requested tolerance exceeds that which is allowed by the given spatial partitioning.")

        if not self.proximity_checks_found:
            self.find_proximity_checks()

        duplicates = []
        keys_list = list(self.points.keys())
        squared_tolerance = tolerance * tolerance
        for points_to_check in self.points_to_check:
            i, j = points_to_check
            p1 = self.points[i]
            p2 = self.points[j]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            test_dist = dx * dx + dy * dy
            if test_dist < squared_tolerance:
                duplicates.append((keys_list.index(i), keys_list.index(j)))
        return duplicates

    def get_line_partitions(self, p1x, p1y, p2x, p2y):
        # This is based on Amanatides & Woo's algorithm as described here:
        # https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/master/overview/FastVoxelTraversalOverview.md
        # It should be noted that this implementation assumes that the line segment begins on the spatial grid
        # which simplifies the initialization of some of the values.

        buckets_visited = []

        p1x_bucket, p1y_bucket = self._get_bucket_key(p1x, p1y)
        p2x_bucket, p2y_bucket = self._get_bucket_key(p2x, p2y)

        dx = p2x - p1x
        dy = p2y - p1y

        if dx == 0:
            step_x = 0
            t_max_x = float("inf")
        elif dx > 0:
            step_x = 1
            t_max_x = ((p1x_bucket + 1) * self.spacing - p1x) / dx
        else:
            step_x = -1
            t_max_x = (p1x_bucket * self.spacing - p1x) / dx

        if dy == 0:
            step_y = 0
            t_max_y = float("inf")
        elif dy > 0:
            step_y = 1
            t_max_y = ((p1y_bucket + 1) * self.spacing - p1y) / dy
        else:
            step_y = -1
            t_max_y = (p1y_bucket * self.spacing - p1y) / dy

        t_delta_x = self.spacing / abs(dx) if dx != 0 else float("inf")
        t_delta_y = self.spacing / abs(dy) if dy != 0 else float("inf")

        current_x_bucket = p1x_bucket
        current_y_bucket = p1y_bucket
        while True:
            buckets_visited.append((current_x_bucket, current_y_bucket))
            if current_x_bucket == p2x_bucket and current_y_bucket == p2y_bucket:
                break
            if t_max_x < t_max_y:
                current_x_bucket += step_x
                t_max_x += t_delta_x
            else:
                current_y_bucket += step_y
                t_max_y += t_delta_y
        return buckets_visited

    def points_close_to_line(self, p1, p2, tolerance):
        if tolerance > self.spacing:
            raise ValueError("Requested tolerance exceeds that which is allowed by the given spatial partitioning.")

        p1x, p1y = p1
        p2x, p2y = p2

        # Trace path of line segment through partitioned space
        buckets_to_check = self.get_line_partitions(p1x, p1y, p2x, p2y)

        # Add in neighboring regions
        neighbors_to_check = []
        for bucket in buckets_to_check:
            for key_modifier in NEIGHBORHOOD_OFFSETS:
                other_bucket = (bucket[0] + key_modifier[0], bucket[1] + key_modifier[1])
                neighbors_to_check.append(other_bucket)

        # Return points that might be too close.
        keys_list = list(self.points.keys())
        buckets_to_check.extend(neighbors_to_check)
        buckets_to_check = set(buckets_to_check)
        close_points = []
        for bucket in buckets_to_check:
            for elem in self.buckets.get(bucket, []):
                close_points.append(keys_list.index(elem))

        return close_points


def gen_shape(prop_bound: list[list[float]], ng_zones=None):
    """Returns an array of shapes objects representing the coordinates given"""
    r_a: list[Shapes | list[Shapes] | None] = [Shapes(prop_bound)]
    if ng_zones is not None:
        r_n = []
        for ng_zone in ng_zones:
            r_n.append(Shapes(ng_zone))
        r_a.append(r_n)
    else:
        r_a.append(None)
    return r_a


def field_optimization_wp_space_fr(
    p_space,
    space_start,
    rotate_step,
    prop_bound,
    ng_zones=None,
    rotate_start=None,
    rotate_stop=None,
    partition_ratio=1.0,
    duplicate_spacing_ratio=0.1,
):
    """Optimizes a Field by iterating over input values w/o perimeter spacing

    Parameters: p_space(float): Ratio of perimeter spacing to other target spacing space_start(float): the initial
    target spacing that the optimization program will start with rotate_step(float): the amount of rotation that will
    be changed per step (in degrees) prop_bound([[float,float]]): 2d array of floats that represent the property
    boundary (counterclockwise) ng_zones([[[float,float]]]): 3d array representing the different zones on the
    property where no boreholes can be placed rotate_start(float): the rotation that the field will start at (-pi/2 <
    rotateStart < pi/2) rotate_stop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop
    < pi/2)

    Outputs: CSVs containing the coordinates for the max field for each target spacing, their respective graphs,
    and their respective data

    """
    if rotate_start is None:
        rotate_start = (-90.0 + rotate_step) * DEG_TO_RAD
    if rotate_stop is None:
        rotate_stop = PI_OVER_2
    if rotate_start > PI_OVER_2 or rotate_start < -PI_OVER_2 or rotate_stop > PI_OVER_2 or rotate_stop < -PI_OVER_2:
        raise ValueError("Invalid Rotation")

    space = space_start
    rt = rotate_start

    y_s = space
    x_s = y_s

    max_l = 0
    max_hole = None
    max_rt = None

    while rt < rotate_stop:
        hole = two_space_gen_bhc(
            prop_bound,
            y_s,
            x_s,
            rotate=rt,
            no_go=ng_zones,
            p_space=p_space * x_s,
            intersection_tolerance=1e-5,
            duplicate_spacing_ratio=duplicate_spacing_ratio,
            partition_ratio=partition_ratio,
        )

        # Assuming that the rotation with the maximum number of boreholes is most efficiently using space
        if hole.size() > max_l:
            max_l = hole.size()
            max_rt = rt * RAD_TO_DEG
            max_hole = hole

        rt += rotate_step * DEG_TO_RAD

    # Ensures that there are no repeated boreholes
    if duplicate_spacing_ratio != 0:
        max_hole = remove_duplicates(max_hole, duplicate_spacing_ratio * max(x_s, y_s, p_space * x_s))

    field = max_hole
    field_name = f"P{p_space:0.1f}_S{space:0.1f}_rt{max_rt:0.1f}"
    return [field.toarray(), field_name]


def field_optimization_fr(
    space_start,
    rotate_step,
    prop_bound,
    ng_zones=None,
    rotate_start=None,
    rotate_stop=None,
    intersection_tolerance=1e-5,
    partition_ratio=1.0,
    duplicate_spacing_ratio=0.1,
):
    """Optimizes a Field by iterating over input values w/o perimeter spacing

    Parameters: space_start(float): the initial target spacing that the optimization program will start with
    rotate_step(float): the amount of rotation that will be changed per step (in degrees) prop_bound([[float,
    float]]): 2d array of floats that represent the property boundary (counterclockwise) ng_zones([[[float,
    float]]]): 3d array representing the different zones on the property where no boreholes can be placed
    rotate_start(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2) rotate_stop(float):
    the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2) intersection_tolerance:

    Outputs: CSVs containing the coordinates for the max field for each target spacing, their respective graphs,
    and their respective data

    """
    if rotate_start is None:
        rotate_start = -90.0 * DEG_TO_RAD
    if rotate_stop is None:
        rotate_stop = PI_OVER_2
    if rotate_start > PI_OVER_2 or rotate_start < -PI_OVER_2 or rotate_stop > PI_OVER_2 or rotate_stop < -PI_OVER_2:
        raise ValueError("Invalid Rotation")

    # Target Spacing iterates

    space = space_start
    rt = rotate_start

    y_s = space
    x_s = y_s

    max_l = 0
    max_hole = None
    max_rt = None

    while rt < rotate_stop:
        hole = gen_borehole_config(
            prop_bound,
            y_s,
            x_s,
            rotate=rt,
            no_go=ng_zones,
            intersection_tolerance=intersection_tolerance,
            duplicate_spacing_ratio=duplicate_spacing_ratio,
            partition_ratio=partition_ratio,
        )

        # Assuming that the rotation with the maximum number of boreholes is most efficiently using space
        """
        print(rt)
        import matplotlib.pyplot as plt
        x_vals = [hole.points[point][0] for point in hole.points]
        y_vals = [hole.points[point][1] for point in hole.points]
        plt.scatter(x_vals, y_vals)
        plt.show()
        """
        if hole.size() > max_l:
            max_l = hole.size()
            max_rt = rt * RAD_TO_DEG
            max_hole = hole

        rt += rotate_step * DEG_TO_RAD

    # Ensures that there are no repeated boreholes
    if duplicate_spacing_ratio != 0:
        max_hole = remove_duplicates(max_hole, duplicate_spacing_ratio * max(x_s, y_s))

    field = max_hole
    field_name = f"S_{space:0.1f}_rt{max_rt:0.1f}"
    return [field.toarray(), field_name]


def sum_sq_dist(p1, p2):
    """Returns the **sum of squared** cartesian distance between two points"""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def pts_dist(p1, p2):
    """Returns the cartesian distance between two points"""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_duplicates(borefield: DeferredDuplicateCheckList, space, disp=False):
    """
    Removes all the duplicates found from the duplicate pairs returned in
    :func:`check_duplicates`.
    For each pair of duplicates, the first borehole (with the lower index) is
    kept and the other (with the higher index) is removed.
    Parameters
    ----------
    borefield : DeferredDuplicateCheckList
        A list of :class:`Borehole` objects
    space:
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    Returns
    -------
    new_borefield : DeferredDuplicateCheckList
        A boreField without duplicates
    """
    # get a list of tuple
    original_size = borefield.size()
    duplicate_pairs = borefield.find_duplicates(space)

    # values not to be included
    duplicate_bores = []
    for i in range(len(duplicate_pairs)):
        duplicate_bores.append(duplicate_pairs[i][1])

    duplicate_bores = sorted(set(duplicate_bores), reverse=True)

    for index in duplicate_bores:
        borefield.delete(index)
    if disp:
        # pad with '-' align in center
        print(
            f"{'*gt.boreholes.remove_duplicates()*':-^50}".replace("*", " ")
        )  # keep a space between the function name
        n_duplicates = original_size - borefield.size()
        print(f"The number of duplicates removed: {n_duplicates}")

    return borefield


def two_space_gen_bhc(
    field,
    y_space,
    x_space,
    no_go=None,
    rotate=0,
    p_space=None,
    i_space=None,
    intersection_tolerance=1e-5,
    duplicate_spacing_ratio=0.1,
    partition_ratio=1.0,
) -> np.array:
    """Generates a borefield that has perimeter spacing

    Parameters:
        field: The outer boundary of the property represented as an array of points
        y_space: Target Spacing in y-dir
        x_space: Target Spacing in x-dir
        no_go: a 3d array representing all the areas where boreholes cannot be placed
        rotate: the amount of rotation (rad)
        p_space: Perimeter spacing
        i_space: Spacing required between perimeter and the rest
        i_space: Min spacing required from all edges
        intersection_tolerance:
        duplicate_spacing_ratio: Used for spatial partitioning (likely does not need to be modified).

    """
    if p_space is None:
        p_space = 0.9 * x_space
    if i_space is None:
        i_space = x_space

    # calls the standard row-wise coord generator w/ the adjusted vertices
    holes = gen_borehole_config(
        field,
        y_space,
        x_space,
        no_go=no_go,
        rotate=rotate,
        intersection_tolerance=intersection_tolerance,
        duplicate_spacing_ratio=duplicate_spacing_ratio,
        partition_ratio=partition_ratio,
    )

    remove_points_too_close(field, holes, i_space, no_go_zones=no_go)

    # places the boreholes along the perimeter of the property boundary and no_go zone(s)
    perimeter_distribute(field, p_space, holes)
    if no_go is not None:
        for ng in no_go:
            perimeter_distribute(ng, p_space, holes)
    holes.partition()
    for i in range(holes.size() - 1, -1, -1):
        point = holes.index(i)
        if point_polygon_check(field.c, point, on_edge_tolerance=1e-3) == -1:
            holes.delete(i)
            continue
        if no_go is not None:
            for ng in no_go:
                if point_polygon_check(ng.c, point, on_edge_tolerance=1e-3) == 1:
                    holes.delete(i)
                    break

    return holes


def remove_points_too_close(field, holes, i_space, no_go_zones=None):
    """
    Will remove all points too close to the field and no-go zones

    Parameters:
        field: The outer boundary of the property represented as an array of points
        no_go_zones: a 3d array representing all the areas where boreholes cannot be placed
        holes: 2d array containing all the current boreholes
        i_space: Min spacing required from all edges
    """
    field = field.c
    len_field = len(field)
    for i in range(len_field):
        p1 = field[i]
        p2 = field[0] if i == len_field - 1 else field[i + 1]
        remove_points_close_to_line(p1, p2, holes, i_space)
    if no_go_zones is not None:
        len_no_go_zones = len(no_go_zones)
        for i in range(len_no_go_zones):
            ng = no_go_zones[i].c
            len_ng = len(ng)
            for j in range(len_ng):
                p1 = ng[j]
                p2 = ng[0] if j == len_ng - 1 else ng[j + 1]
                remove_points_close_to_line(p1, p2, holes, i_space)


def remove_points_close_to_line(p1, p2, holes, i_space):
    """Removes points that are close to the given line

    Parameters:
        p1([float,float]): first point in line
        p2([float,float]): second point in line
        holes: 2d array containing a bunch of points
        i_space(float): distance cutoff for how close points can be

    """
    points_to_check = holes.points_close_to_line(p1, p2, i_space)
    points_to_check = sorted(set(points_to_check), reverse=True)
    for point in points_to_check:
        hole = holes.index(point)
        dp = dist_from_line(p1, p2, hole)
        if dp < i_space:
            holes.delete(point)


def dist_from_line(p1, p2, other_point):
    """Calculates the distance from a point to a line (closest distance):
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameter:
        p1: first point on the line
        p2: second point on the line
        other_point: point which is being measured to

    """
    dxl = p2[0] - p1[0]
    dyl = p2[1] - p1[1]
    dx = p1[0] - other_point[0]
    dy = p1[1] - other_point[1]
    num = abs(dxl * dy - dx * dyl)
    den = sqrt(dxl * dxl + dyl * dyl)
    dp = num / den
    dist_l = pts_dist(p1, p2)
    d01 = pts_dist(p1, other_point)
    d02 = pts_dist(p2, other_point)
    if d01 * d01 - dp * dp < 0:
        return d01
    if sqrt(d01 * d01 - dp * dp) / dist_l > 1:
        return min(d01, d02)
    if min(p1[0], p2[0]) < other_point[0] < max(p1[0], p2[0]) or min(p1[1], p2[1]) < other_point[1] < max(p1[1], p2[1]):
        return min(d01, d02, dp)
    else:
        return min(d01, d02)


def perimeter_distribute(field, space, r):
    """
    Distributes boreholes along the perimeter of a given shape

    Parameters:
        field: array of points representing closed polygon
        space (float): spacing that the boreholes should have from one another
        r (dict{}): existing dictionary of boreholes which will be appended to
    """
    total_arc_length = field.total_arc_length
    act_number_of_boreholes = int(total_arc_length // space)
    perimeter_spacing_ratios = np.linspace(0.0, 1.0, act_number_of_boreholes, endpoint=False)
    for ratio in perimeter_spacing_ratios:
        r.append(field.get_point_along_curve(ratio))


def gen_borehole_config(
    field,
    y_space,
    x_space,
    no_go=None,
    rotate=0,
    intersection_tolerance=1e-6,
    duplicate_spacing_ratio=0.1,
    partition_ratio=1.0,
):
    """
    Function generates a series of x,y points representing a field of boreholes
    in a trapezoidal shape. Returns empty if boreHole field does not meet given requirements

    Parameters
    -------------
        :param field:
        :param y_space: float
            the minimum spacing between points in the y-dir
        :param x_space: float
            the minimum spacing between points in the x-dir
        :param no_go:
        :param rotate:
        :param intersection_tolerance:
        :param duplicate_spacing_ratio:
        :param partition_ratio: float
            Defines the size of the space partitioning

        :return: [[float]] -> 2 col + n rows
    """

    if no_go is None:
        no_go = []
    # Decides which vertex to start generating boreholes at by finding the "lowest" vertex relative to a rotated x-axis
    lowest_vert_val = inf
    highest_vert_val = -inf
    lowest_vert = None
    highest_vert = None
    for vert in field.c:
        phi = atan(vert[1] / vert[0]) if vert[0] != 0 else PI_OVER_2
        dist_vert = sqrt(vert[1] ** 2 + vert[0] ** 2)
        ref_angle = phi
        if phi > PI_OVER_2:
            if phi > pi:
                ref_angle = 2 * rotate + 3 * PI_OVER_2 - phi if phi > 3 * PI_OVER_2 else 2 * rotate + pi - phi
            else:
                ref_angle = pi - phi + 2 * rotate
        yp = dist_vert * sin(ref_angle - rotate)
        if yp < lowest_vert_val:
            lowest_vert_val = yp
            lowest_vert = vert
        if yp > highest_vert_val:
            highest_vert_val = yp
            highest_vert = vert

    # Determines the number of rows as well as the distance between the rows
    num_rows = int((highest_vert_val - lowest_vert_val) // y_space)
    d = highest_vert_val - lowest_vert_val
    s = d / num_rows
    row_space = [-1 * s * cos(PI_OVER_2 - rotate), s * sin(PI_OVER_2 - rotate)]

    # Establishes the list object
    boreholes = DeferredDuplicateCheckList(partition_ratio * max(x_space, y_space))
    row_point = [lowest_vert[0], lowest_vert[1]]

    # This is just a value that is combined with the slope of the row's to establish two points defining a row (could
    # be any value)
    point_shift = 1000.0

    for _ in range(num_rows + 1):
        # Row Defined by two points
        if row_space[1] == 0:
            row = [
                row_point[0],
                row_point[1],
                row_point[0],
                row_point[1] + point_shift,
            ]
        else:
            row = [
                row_point[0],
                row_point[1],
                row_point[0] + point_shift,
                row_point[1] + (-row_space[0] / row_space[1]) * point_shift,
            ]

        # Gets Intersection between current row and property boundary
        f_inters = field.line_intersect(row, rotate, intersection_tolerance)

        # Stores the number of intersections with the row
        len_f_inters = len(f_inters)

        # Checks for edge case where a single intersection is reported as two and treats it as one
        if (
            len_f_inters > 1
            and abs(f_inters[0][0] - f_inters[1][0]) <= intersection_tolerance
            and abs(f_inters[0][1] - f_inters[1][1]) <= intersection_tolerance
        ):
            fi = 0
            fij = 0
            while fi < len_f_inters:
                while fij < len_f_inters:
                    if fij == fi:
                        fij += 1
                        continue
                    if (
                        abs(f_inters[fi][0] - f_inters[fij][0]) <= intersection_tolerance
                        and abs(f_inters[fi][1] - f_inters[fij][1]) <= intersection_tolerance
                    ):
                        f_inters.pop(fij)
                        if fi >= fij:
                            fi -= 1
                        fij -= 1
                        len_f_inters -= 1
                    fij += 1
                fi += 1

        # Checks for edge case where there are no intersections detected due to a rounding error (can sometimes
        # happen with the last row)
        """

        if len_f_inters == 0 and ri == num_rows:
            ins = False

            # Checks if the predicted point (ghost point that was expected but not found) is inside one of the no_go
            zones for shape in no_go: if shape.point_intersect(highest_vert): ins = True if not ins: #Double checks
            that this borehole has not already been included if len(boreholes)==0 or not (boreholes[len(boreholes) -
            1][0] == highest_vert[0] and boreholes[len(boreholes) - 1][1] ==highest_vert[1]): boreholes[len(
            boreholes)] = highest_vert """
        # Handles cases with odd number of intersections
        if len_f_inters % 2 == 0:
            # Specific case with two intersections
            if len_f_inters == 2:  # noqa: PLR2004, SIM102
                # Checks for the edge case where two intersections are very close together and replaces them with one
                # point
                if (
                    sqrt(
                        (f_inters[0][0] - f_inters[1][0]) * (f_inters[0][0] - f_inters[1][0])
                        + (f_inters[0][1] - f_inters[1][1]) * (f_inters[0][1] - f_inters[1][1])
                    )
                    < x_space
                ):
                    ins = False
                    for ng_shape in no_go:
                        if ng_shape.point_intersect(highest_vert):
                            ins = True
                    if not ins:
                        boreholes.append(f_inters[0])
                        len_f_inters = 0  # skips the while loop

            i = 0
            while i < len_f_inters - 1:
                left_offset = [0, 0]
                right_offset = [0, 0]

                # Checks if there is enough distance between this point and another and then will offset the point if
                # there is not enough room
                dls_check = sqrt(
                    (f_inters[i][0] - f_inters[i - 1][0]) * (f_inters[i][0] - f_inters[i - 1][0])
                    + (f_inters[i][1] - f_inters[i - 1][1]) * (f_inters[i][1] - f_inters[i - 1][1])
                )

                drs_check = sqrt(
                    (f_inters[i][0] - f_inters[i + 1][0]) * (f_inters[i][0] - f_inters[i + 1][0])
                    + (f_inters[i][1] - f_inters[i + 1][1]) * (f_inters[i][1] - f_inters[i + 1][1])
                )

                if i > 0 and (dls := dls_check) < x_space:
                    left_offset = [dls * cos(rotate), dls * sin(rotate)]
                elif i < len_f_inters - 1 and (drs := drs_check) < x_space:
                    right_offset = [-drs * cos(rotate), -drs * sin(rotate)]

                process_rows(
                    row,
                    [f_inters[i][0] + left_offset[0], f_inters[i][1] + left_offset[1]],
                    [
                        f_inters[i + 1][0] + right_offset[0],
                        f_inters[i + 1][1] + right_offset[1],
                    ],
                    no_go,
                    x_space,
                    boreholes,
                    rotate=rotate,
                )

                i += 2
        elif len_f_inters == 1:
            ins = False
            for ng_shape in no_go:
                if ng_shape.point_intersect(highest_vert):
                    ins = True
            if not ins:
                if boreholes.size() == 0:
                    boreholes.append(f_inters[0])
                if not (
                    boreholes.index(boreholes.size() - 1)[0] == f_inters[0][0]
                    and boreholes.index(boreholes.size() - 1)[1] == f_inters[0][1]
                ):
                    boreholes.append(f_inters[0])
        else:
            i = 0
            while i < len_f_inters - 1:
                if field.point_intersect(
                    [
                        (f_inters[i][0] + f_inters[i + 1][0]) / 2,
                        (f_inters[i][1] + f_inters[i + 1][1]) / 2,
                    ]
                ):
                    process_rows(
                        row,
                        f_inters[i],
                        f_inters[i + 1],
                        no_go,
                        x_space,
                        boreholes,
                        rotate=rotate,
                    )
                i += 1
        row_point[0] += row_space[0]
        row_point[1] += row_space[1]
    borefield = remove_duplicates(boreholes, duplicate_spacing_ratio * max(x_space, y_space))
    return borefield


def process_rows(row, row_sx, row_ex, no_go, row_space, r_a, rotate, intersection_tolerance=1e-5):
    """
    Function generates a row of the borefield
    *Note: the formatting from the rows can be a little unexpected. Some adjustment
    may be required to correct the formatting. The genBoreHoleConfig function already accounts for this.
    Parameters
    -------------
    :param row:
    :param row_sx:
    :param row_ex:
    :param no_go:
    :param row_space:
    :param r_a:
    :param rotate:
    :param intersection_tolerance:
    """

    if no_go is None:
        distribute(row_sx, row_ex, row_space, r_a)
        return r_a

    # row = [row_sx[0], row_sx[1], row_ex[0], row_ex[1]]

    num_col = int(
        sqrt((row_sx[0] - row_ex[0]) * (row_sx[0] - row_ex[0]) + (row_sx[1] - row_ex[1]) * (row_sx[1] - row_ex[1]))
        // row_space
    )

    inters = [
        point
        for shape in no_go
        for point in shape.line_intersect(row, rotate=rotate, intersection_tolerance=intersection_tolerance)
    ]
    inters = sort_intersections(inters, rotate)
    num_inters = len(inters)

    if num_inters > 1:  # noqa: SIM102
        if less_than(
            inters[0],
            row_sx,
            rotate=rotate,
            intersection_tolerance=intersection_tolerance,
        ) and less_than(
            row_ex,
            inters[len(inters) - 1],
            rotate=rotate,
            intersection_tolerance=intersection_tolerance,
        ):
            inside = False
            for _ in inters:
                if less_than(
                    row_sx,
                    inters[0],
                    rotate=rotate,
                    intersection_tolerance=intersection_tolerance,
                ) and less_than(
                    inters[0],
                    row_ex,
                    rotate=rotate,
                    intersection_tolerance=intersection_tolerance,
                ):
                    inside = True
            if not inside:
                point_in = False
                for ng_shape in no_go:
                    if ng_shape.point_intersect([(row_ex[0] + row_sx[0]) / 2, (row_ex[1] + row_sx[1]) / 2]):
                        point_in = True
                if point_in:
                    return []
    inters = np.array(inters)
    indices = []
    for j in range(num_inters):
        less_than_1 = less_than(row_ex, inters[j], rotate=rotate, intersection_tolerance=intersection_tolerance)
        less_than_2 = less_than(inters[j], row_sx, rotate=rotate, intersection_tolerance=intersection_tolerance)
        if not (less_than_1 or less_than_2):
            indices.append(j)
    inters = inters[indices]
    num_inters = len(inters)
    for i in range(num_inters - 1):
        space = sqrt(
            (inters[i + 1][0] - inters[i][0]) * (inters[i + 1][0] - inters[i][0])
            + (inters[i + 1][1] - inters[i][1]) * (inters[i + 1][1] - inters[i][1])
        )
        if space < row_space:
            i_none = False
            for shape in no_go:
                if shape.point_intersect(
                    [
                        (inters[i + 1][0] + inters[i][0]) / 2,
                        (inters[i + 1][1] + inters[i][1]) / 2,
                    ]
                ):
                    i_none = True
            if i_none:
                d = (row_space - space) / 2
                inters[i + 1][0] += d * cos(rotate)
                inters[i + 1][1] += d * sin(rotate)
                inters[i][0] -= d * cos(rotate)
                inters[i][1] -= d * sin(rotate)
    if num_col < 1:
        ins = False
        for shape in no_go:
            if shape.point_intersect([(row_ex[0] + row_sx[0]) / 2, (row_ex[1] + row_sx[1]) / 2]):
                ins = True
        if not ins:
            if r_a.size() == 0 or not (
                r_a.index(r_a.size() - 1)[0] == (row_ex[0] + row_sx[0]) / 2
                and r_a.index(r_a.size() - 1)[1] == (row_ex[1] + row_sx[1]) / 2
            ):
                r_a.append([(row_ex[0] + row_sx[0]) / 2, (row_ex[1] + row_sx[1]) / 2])
            return r_a
    elif num_inters == 0:
        if not_inside(row_sx, no_go) and not_inside(row_ex, no_go):
            distribute(row_sx, row_ex, row_space, r_a)
    elif num_inters == 2:  # noqa: PLR2004
        distribute(row_sx, inters[0], row_space, r_a)
        distribute(inters[1], row_ex, row_space, r_a)
    elif num_inters == 1:
        ins = False
        for shape in no_go:
            if shape.point_intersect([(inters[0][0] + row_sx[0]) / 2, (inters[0][1] + row_sx[1]) / 2]):
                ins = True
        if not ins:
            distribute(row_sx, inters[0], row_space, r_a)
        else:
            distribute(inters[0], row_ex, row_space, r_a)
    elif num_inters % 2 == 0:
        i = 0
        while i < num_inters:
            if i == 0:
                distribute(row_sx, inters[0], row_space, r_a)
                i = 1
                continue
            elif i == num_inters - 1:
                distribute(inters[num_inters - 1], row_ex, row_space, r_a)
            else:
                distribute(inters[i], inters[i + 1], row_space, r_a)
            i += 2
    else:
        ins = False
        for shape in no_go:
            if shape.point_intersect([(inters[0][0] + row_sx[0]) / 2, (inters[0][1] + row_sx[1]) / 2]):
                ins = True
        if not ins:
            i = 0
            while i < num_inters:
                if i == 0:
                    distribute(row_sx, inters[0], row_space, r_a)
                    i = 1
                    continue
                elif i == num_inters - 1:
                    i += 2
                    continue
                else:
                    distribute(inters[i], inters[i + 1], row_space, r_a)
                i += 2
        else:
            i = 0
            while i < num_inters:
                if i == 0:
                    distribute(inters[0], inters[1], row_space, r_a)
                    i = 2
                    continue
                elif i == num_inters - 1:
                    distribute(inters[i], row_ex, row_space, r_a)
                    i += 2
                    continue
                else:
                    distribute(inters[i], inters[i + 1], row_space, r_a)
                i += 2

    return r_a


def not_inside(p, ngs):
    inside = False
    for ng in ngs:
        if ng.point_intersect(p):
            inside = True
    return not inside


def less_than(p1, p2, rotate=0, intersection_tolerance=1e-5):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < intersection_tolerance:
        dx_sign = 0
    elif dx > 0:
        dx_sign = 1
    else:
        dx_sign = -1

    if abs(dy) < intersection_tolerance:
        dy_sign = 0
    elif dy > 0:
        dy_sign = 1
    else:
        dy_sign = -1

    if rotate >= 0:
        if dx_sign == 0:
            return dy_sign == 1
        elif dy_sign in (0, dx_sign):
            return dx_sign == 1
        else:
            raise ValueError("Slope between points does not match field orientation.")
    elif dx_sign == 0:
        if dy_sign == 1:
            return False
        else:
            return False
    elif dy_sign == 0 or dx_sign != dy_sign:
        return dx_sign == 1
    else:
        raise ValueError("Slope between points does not match field orientation.")


@lru_cache(maxsize=1024)
def evenly_spaced_points(number_of_points, starting_val=0.0, ending_val=1.0):
    return np.linspace(starting_val, ending_val, num=number_of_points, endpoint=True)


def distribute(x1, x2, spacing, r):
    """
      Function generates a series of boreholes between x1 and x2
    Parameters
    -------------
    :param x1: float
        left x value
    :param x2: float
        right x value
    :param spacing: float
        spacing between columns
    :param r: [[float]]
        existing array of points
    :param rotate:
    :return:
    """
    # This is the squared distance between the two row edges.
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    d = sqrt(dx * dx + dy * dy)

    # Check if there is only room for one point along the row
    if d < spacing:
        # If we have currently not generated any coordinates, or we have not added a point at the same location
        # previously, then add at the average between the two row edges.
        if r.size() == 0 or not (
            r.index(r.size() - 1)[0] == (x1[0] + x2[0]) / 2 and r.index(r.size() - 1)[1] == (x1[1] + x2[1]) / 2
        ):
            r.append([(x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2])
        return r

    # Determine the number of boreholes placed along the row.
    act_num_col = int(d // spacing) + 1

    # Determine the x and y values along the row
    spacing_multipliers = evenly_spaced_points(act_num_col)
    x_vals = x1[0] + dx * spacing_multipliers
    y_vals = x1[1] + dy * spacing_multipliers
    r.extend(zip(x_vals, y_vals))
    return r
