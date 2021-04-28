"""MS MARCO Data download and processing
"""

import os
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--datapath', default='/data',
    help = 'Directory where the raw data is present')

args = parser.parse_args()



