from gaitnet.gaitnet import train
from gaitnet.util import log_exceptions
from gaitnet import get_logger
logger = get_logger()

if __name__ == "__main__":
    from gaitnet.util import log_exceptions
    with log_exceptions(logger):
        train.main()