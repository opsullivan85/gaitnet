from gaitnet.util import log_exceptions
from gaitnet import get_logger
logger = get_logger()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ContactNet Main Entry Point")
    # data-gen
    parser.add_argument("--datagen", action="store_true", help="Generate data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--data-info", action="store_true", help="Find data files")
    parser.add_argument("--dfs-debug", action="store_true", help="Run DFS debug")
    parser.add_argument("--manual-control", action="store_true", help="Run robot with controller and network")
    args = parser.parse_known_args()
    used_args, unknown_args = args

    with log_exceptions(logger):
        if used_args.datagen:
            from gaitnet.contactnet import datagen
            datagen.main()

        elif used_args.train:
            from gaitnet.contactnet import contactnet
            contactnet.main()

        elif used_args.evaluate:
            from gaitnet.contactnet import evaluate
            evaluate.main()
        
        elif used_args.dfs_debug:
            from gaitnet.contactnet import datagen
            datagen.dfs_debug()

        elif used_args.data_info:
            from gaitnet.contactnet import datainfo
            datainfo.main()
        
        elif used_args.manual_control:
            from gaitnet.contactnet import manualcontrol
            manualcontrol.main()
        
        else:
            raise ValueError("No valid arguments provided. Use --help for more information.")