from pygfunction.boreholes import Borehole

from ghedesigner.enums import BHPipeType, DoubleUTubeConnType
from ghedesigner.ghe.boreholes.coaxial_borehole import CoaxialPipe
from ghedesigner.ghe.boreholes.multi_u_borehole import MultipleUTube
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil


def get_bhe_object(
    bhe_type: BHPipeType,
    m_flow_borehole: float,
    fluid: GHEFluid,
    _borehole: Borehole,
    pipe: Pipe,
    grout: Grout,
    soil: Soil,
):
    if bhe_type == BHPipeType.SINGLEUTUBE:
        return SingleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil)
    elif bhe_type == BHPipeType.DOUBLEUTUBEPARALLEL:
        return MultipleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil, config=DoubleUTubeConnType.PARALLEL)
    elif bhe_type == BHPipeType.DOUBLEUTUBESERIES:
        return MultipleUTube(m_flow_borehole, fluid, _borehole, pipe, grout, soil, config=DoubleUTubeConnType.SERIES)
    elif bhe_type == BHPipeType.COAXIAL:
        return CoaxialPipe(m_flow_borehole, fluid, _borehole, pipe, grout, soil)
    else:
        raise TypeError("BHE type not implemented")
