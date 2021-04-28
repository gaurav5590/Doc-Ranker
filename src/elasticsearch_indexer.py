"""Indexing all the documents for querying later
"""

import os
import re
import argparse
import elasticsearch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--xx', default='/data',
    help = 'Directory where the raw data is present')

args = parser.parse_args()