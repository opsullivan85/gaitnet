from gaitnet.util import log_exceptions

from gaitnet import get_logger
logger = get_logger()
from gaitnet.simulation.simulation import main

if __name__ == "__main__":
    with log_exceptions(logger):
        main()
