import argparse
from typing import Tuple


def get_arguments() -> Tuple[str, int]:
    """
    Parse name of file with urls and number of threads from command line.
    Number of threads is optional argument and by default equals 1.

    :return: name of file with urls and number of thread tuple
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="file with a list of references", type=str)
    parser.add_argument(
        "-t",
        "--threads",
        help="set the number of threads to the given number",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    return args.file_name, args.threads
