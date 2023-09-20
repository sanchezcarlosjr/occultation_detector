import argparse
import logging
import sys
import os

from occultation_detector import __version__
from occultation_detector.server import launch_web_server
from occultation_detector.pipeline import NpyFileLoader, pipeline
import json
import dataclasses
import pandas as pd

__author__ = "sanchezcarlosjr"
__copyright__ = "sanchezcarlosjr"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


class WebSimulatorAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        _logger.debug("Starting...")
        launch_web_server()
        _logger.debug("Server is up")

class PredictorAction(argparse.Action):
    def __call__(self, parser, namespace, files, option_string=None):
        _logger.debug("Starting...")
        predict = pipeline('checkpoints/vanilla-neuronal-network.keras')
        ds = NpyFileLoader(files[0].name)
        result = predict(ds)
        print(result)
        _logger.debug("Prediction has been done")

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Detect and Simulate Serendipitous Stellar Occultations")
    parser.add_argument(
        "--version",
        action="version",
        version=f"occultation_detector {__version__}",
    )
    parser.add_argument(
       '-p', 
       '--predict', 
       nargs=1,
       type=argparse.FileType('r'),
       default=sys.stdin,
       action=PredictorAction,
       help='predict the features of a Trans-Neptunian Object (TNO) using a CSV file containing light curve data, which includes timestamps and corresponding intensity values.',
    )
    parser.add_argument(
       '-pl', 
       '--plot', 
       help='plot the light curve data using a CSV file, which includes timestamps and corresponding intensity values.'
    )
    parser.add_argument(
        "-s",
        "--server",
        dest="server", 
        help="Launch web simulator", 
        type=str, 
        action=WebSimulatorAction
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    _logger.debug("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
