
from gaitnet import get_logger
logger = get_logger()

logger.debug("initialized")

from gaitnet.sim2real.siminterface import SimInterface
from gaitnet.sim2real.abstractinterface import Sim2RealInterface
from gaitnet.sim2real.realinterface import RealInterface