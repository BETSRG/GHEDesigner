from enum import IntEnum, auto
from json import loads
from pathlib import Path

class Field(IntEnum):
    NAME = 0
    MONTHLY_HTG = auto()
    MONTHLY_CLG = auto()
    PK_HTG = auto()
    PK_HTG_DUR = auto()
    PK_CLG = auto()
    PK_CLG_DUR = auto()


class Month(IntEnum):
    JAN = 0
    FEB = auto()
    MAR = auto()
    APR = auto()
    MAY = auto()
    JUN = auto()
    JUL = auto()
    AUG = auto()
    SEP = auto()
    OCT = auto()
    NOV = auto()
    DEC = auto()


def get_max_monthly(data: dict, field: Field) -> int:
    return round(max([
        data[Month.JAN][field],
        data[Month.FEB][field],
        data[Month.MAR][field],
        data[Month.APR][field],
        data[Month.MAY][field],
        data[Month.JUN][field],
        data[Month.JUL][field],
        data[Month.AUG][field],
        data[Month.SEP][field],
        data[Month.OCT][field],
        data[Month.NOV][field],
        data[Month.DEC][field],
    ]))

def write_monthly_loads(data: dict) -> str:
    name = data[Field.NAME]
    len_name = len(name)
    monthly_htg = f'{data[Field.MONTHLY_HTG]:0.2f}'
    pad_1 = 23 - len_name - len(monthly_htg)
    monthly_clg = f'{data[Field.MONTHLY_CLG]:0.2f}'
    pad_2 = 18 - len(monthly_clg)
    pk_htg = f'{data[Field.PK_HTG]:0.2f}'
    pad_3 = 18 - len(pk_htg)
    pk_clg = f'{data[Field.PK_CLG]:0.2f}'
    pad_4 = 18 - len(pk_clg)

    s = name + \
        ' ' * pad_1 + \
        monthly_htg + \
        ' ' * pad_2 + \
        monthly_clg + \
        ' ' * pad_3 + \
        pk_htg + \
        ' ' * pad_4 + \
        pk_clg
    return s


def write_glhepro_file(input_file: Path):
    if not input_file.exists():
        print(f"Simulation summary file not found, {input_file}. Aborting.")

    output_dict = loads(input_file.read_text())
    replacement_vals = {
        'borehole_radius': output_dict['ghe_system']['borehole_diameter']['value'] / 2.0,
        'borehole_depth': output_dict['ghe_system']['active_borehole_length']['value'],
        'borehole_spacing': output_dict['ghe_system']['borehole_spacing']['value'],
        'borehole_effective_resistance': output_dict['ghe_system']['effective_borehole_resistance']['value'],
        'pipe_inner_diameter': output_dict['ghe_system']['pipe_geometry']['pipe_inner_diameter']['value'],
        'pipe_outer_diameter': output_dict['ghe_system']['pipe_geometry']['pipe_outer_diameter']['value'],
        'shank_spacing': output_dict['ghe_system']['shank_spacing']['value'],
        'pipe_thermal_conductivity': output_dict['ghe_system']['pipe_thermal_conductivity']['value'],
        'soil_thermal_conductivity': output_dict['ghe_system']['soil_thermal_conductivity']['value'],
        'soil_volumetric_heat_capacity': output_dict['ghe_system']['soil_volumetric_heat_capacity']['value'],
        'soil_undisturbed_ground_temp': output_dict['ghe_system']['soil_undisturbed_ground_temp']['value'],
        'fluid_thermal_conductivity': output_dict['ghe_system']['fluid_thermal_conductivity']['value'],
        'fluid_volumetric_heat_capacity': output_dict['ghe_system']['fluid_volumetric_heat_capacity']['value'],
        'fluid_viscosity': output_dict['ghe_system']['fluid_viscosity']['value'],
        'fluid_density': output_dict['ghe_system']['fluid_density']['value'],
        'grout_thermal_conductivity': output_dict['ghe_system']['grout_thermal_conductivity']['value'],
        'grout_volumetric_heat_capacity': output_dict['ghe_system']['grout_volumetric_heat_capacity']['value'],
        'start_month': output_dict['simulation_parameters']['start_month'],
        'end_month': output_dict['simulation_parameters']['end_month'],
        'maximum_allowable_hp_eft': output_dict['simulation_parameters']['maximum_allowable_hp_eft']['value'],
        'minimum_allowable_hp_eft': output_dict['simulation_parameters']['minimum_allowable_hp_eft']['value'],
        'january_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.JAN]),
        'february_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.FEB]),
        'march_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.MAR]),
        'april_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.APR]),
        'may_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.MAY]),
        'june_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.JUN]),
        'july_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.JUL]),
        'august_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.AUG]),
        'september_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.SEP]),
        'october_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.OCT]),
        'november_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.NOV]),
        'december_glhe_loads': write_monthly_loads(output_dict['ghe_system']['glhe_monthly_loads']['data'][Month.DEC]),
        'peak_heating_hrs': get_max_monthly(output_dict['ghe_system']['glhe_monthly_loads']['data'], Field.PK_HTG_DUR),
        'peak_cooling_hrs': get_max_monthly(output_dict['ghe_system']['glhe_monthly_loads']['data'], Field.PK_CLG_DUR),
    }

    template_path = Path(__file__).parent.parent / 'glhepro' / 'template_v5.1.gli'
    template_file = template_path.read_text()

    out_str = template_file.format(**replacement_vals)

    with open((input_file.parent / "GLHEPro.gli"), "w") as f_out:
        f_out.write(out_str)
