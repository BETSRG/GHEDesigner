from dataclasses import dataclass, field
from typing import TypeGuard, cast

from pygfunction.boreholes import Borehole

from ghedesigner.enums import DesignGeomType, TimestepType
from ghedesigner.ghe.design.base import DesignBase, GeometricConstraints
from ghedesigner.ghe.domains import (
    general_domain_nbh_adjustment,
    polygonal_land_constraint,
    polygonal_land_constraint_multi_field,
)
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.search.bisection_zd import Bisection1D, BisectionZD
from ghedesigner.media import Fluid, Grout, Soil

Coordinate = tuple[float, float]
BoreholeRemovalValue = str | list[Coordinate] | list[list[Coordinate]]


def is_2d(property_boundary: list[list[float]] | list[list[list[float]]]) -> TypeGuard[list[list[float]]]:
    return bool(property_boundary) and isinstance(property_boundary[0][0], (int, float))


@dataclass
class GeometricConstraintsBiRectangleConstrained(GeometricConstraints):
    """
    Geometric constraints for bi-rectangle constrained design algorithm
    """

    b_min: float
    b_max_x: float | None = None
    b_max_y: float | None = None
    property_boundary: list[list[list[float]]] = field(init=False)
    no_go_boundaries: list[list[list[float]]] | None = None
    borehole_removal_options: dict[str, BoreholeRemovalValue] = field(default_factory=dict)
    type: DesignGeomType = field(default=DesignGeomType.BIRECTANGLECONSTRAINED, init=False, repr=False)

    def __init__(
        self,
        b_min: float,
        property_boundary: list[list[float]] | list[list[list[float]]],
        b_max_x: float | None = None,
        b_max_y: float | None = None,
        no_go_boundaries: list[list[list[float]]] | None = None,
        borehole_removal_options: dict[str, BoreholeRemovalValue] | None = None,
    ) -> None:
        self.b_min = b_min
        self.b_max_x = b_max_x
        self.b_max_y = b_max_y
        self.no_go_boundaries = no_go_boundaries
        self.borehole_removal_options = borehole_removal_options or {}

        if is_2d(property_boundary):
            self.property_boundary = [property_boundary]
        else:
            self.property_boundary = cast(list[list[list[float]]], property_boundary)

    def to_input(self) -> dict[str, object]:
        result: dict[str, object] = {
            "method": "bi_rectangle_constrained",
            "minimum_borehole_spacing_m": self.b_min,
            "property_boundary_coordinates_m": [{"x": x, "y": y} for x, y in self.property_boundary[0]],
        }
        if self.b_max_x is not None:
            result["maximum_borehole_spacing_x_m"] = self.b_max_x
        if self.b_max_y is not None:
            result["maximum_borehole_spacing_y_m"] = self.b_max_y
        if self.no_go_boundaries is not None:
            result["no_go_boundary_coordinates_m"] = [
                [{"x": x, "y": y} for x, y in boundary] for boundary in self.no_go_boundaries
            ]
        if self.borehole_removal_options:
            options = dict(self.borehole_removal_options)
            method = str(options.pop("borehole_removal_method", "radial")).lower()
            if method == "linesegments":
                method = "line_segments"
            elif method == "righttop":
                method = "right_top"
            serialized_options: dict[str, object] = {"borehole_removal_method": method}
            if "points" in options:
                points = cast(list[Coordinate], options["points"])
                serialized_options["borehole_removal_points_m"] = [{"x": x, "y": y} for x, y in points]
            if "line_segments" in options:
                line_segments = cast(list[list[Coordinate]], options["line_segments"])
                serialized_options["borehole_removal_line_segments_m"] = [
                    [{"x": x, "y": y} for x, y in segment] for segment in line_segments
                ]
            result["borehole_removal_options"] = serialized_options
        return result


class DesignBiRectangleConstrained(DesignBase):
    def __init__(
        self,
        v_flow: float,
        borehole: Borehole,
        fluid: Fluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        start_month: int,
        end_month: int,
        max_eft: float,
        min_eft: float,
        max_height: float,
        min_height: float,
        continue_if_design_unmet: bool,
        max_boreholes: int | None,
        geometric_constraints: GeometricConstraintsBiRectangleConstrained,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        load_years=None,
        keep_contour: tuple[bool, bool] | None = None,
    ) -> None:
        super().__init__(
            v_flow,
            borehole,
            fluid,
            pipe,
            grout,
            soil,
            start_month,
            end_month,
            max_eft,
            min_eft,
            max_height,
            min_height,
            continue_if_design_unmet,
            max_boreholes,
            geometric_constraints,
            hourly_extraction_ground_loads,
            method,
            load_years,
        )
        if keep_contour is None:
            keep_contour = cast(tuple[bool, bool], [True, False])
        self.geometric_constraints = geometric_constraints
        if self.geometric_constraints.b_max_x is not None and self.geometric_constraints.b_max_y is not None:
            self.coordinates_domain, self.fieldDescriptors = polygonal_land_constraint(
                self.geometric_constraints.b_min,
                self.geometric_constraints.b_max_x,
                self.geometric_constraints.b_max_y,
                self.geometric_constraints.property_boundary,
                self.geometric_constraints.no_go_boundaries,
                keep_contour=keep_contour,
            )
            self.borehole_lengths = [len(coords) for dom in self.coordinates_domain for coords in dom]
            self.min_nbh = min(self.borehole_lengths)
            self.max_nbh = max(self.borehole_lengths)
            self.domain_2d = True
        else:
            self.coordinates_domain, self.fieldDescriptors = polygonal_land_constraint_multi_field(
                self.geometric_constraints.b_min,
                [self.geometric_constraints.property_boundary],
                self.geometric_constraints.no_go_boundaries,
                keep_contour=keep_contour,
                split_domains_by_property=False,
            )
            self.borehole_lengths = [len(coords) for coords in self.coordinates_domain]
            self.domain_2d = False
            self.min_nbh = min(self.borehole_lengths)
            self.max_nbh = max(self.borehole_lengths)

    def find_design(self, disp=False) -> Bisection1D | BisectionZD:
        if disp:
            title = "Find bi-rectangle_constrained..."
            print(title + "\n" + len(title) * "=")
        if self.domain_2d:
            return BisectionZD(
                self.coordinates_domain,
                self.fieldDescriptors,
                self.v_flow,
                self.borehole,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.max_boreholes,
                self.min_height,
                self.max_height,
                self.continue_if_design_unmet,
                self.start_month,
                self.end_month,
                self.min_EFT_allowable,
                self.max_EFT_allowable,
                self.hourly_extraction_ground_loads,
                method=self.method,
                disp=disp,
                field_type="bi-rectangle_constrained",
                load_years=self.load_years,
            )
        else:
            return Bisection1D(
                self.coordinates_domain,
                self.fieldDescriptors,
                self.v_flow,
                self.borehole,
                self.fluid,
                self.pipe,
                self.grout,
                self.soil,
                self.max_boreholes,
                self.min_height,
                self.max_height,
                self.continue_if_design_unmet,
                self.start_month,
                self.end_month,
                self.min_EFT_allowable,
                self.max_EFT_allowable,
                self.hourly_extraction_ground_loads,
                method=self.method,
                disp=disp,
                field_type="bi-rectangle_constrained",
                load_years=self.load_years,
            )

    def get_bounds(self):
        return self.min_nbh, self.max_nbh

    def closest_nbh(self, desired_nbh):
        if self.domain_2d:
            raise ValueError(
                'Only a 1D BUPCRS domain supports the "closest_nbh" function. You can create a 1d domain'
                "by simply omitting the b_max_x and b_max_y inputs to the BUPCRS geometric constraints."
            )
        return general_domain_nbh_adjustment(
            self.coordinates_domain,
            self.borehole_lengths,
            self.min_nbh,
            self.max_nbh,
            desired_nbh,
            self.geometric_constraints.borehole_removal_options,
        )
