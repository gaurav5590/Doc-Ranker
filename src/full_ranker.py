"""Full ranking using Elastic Search BM25
"""

import os
import re
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--topn', default=1000,
    help = 'Top N to be returned after full-ranking')

args = parser.parse_args()

