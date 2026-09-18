from gaitnet.util import log_exceptions
from gaitnet import get_logger, setup_logging
logger = get_logger()

@log_exceptions(logger)
def main():
    setup_logging()
    print("Hello, World!")

if __name__ == "__main__":
    from gaitnet.util import log_exceptions
    with log_exceptions(logger):
        main()