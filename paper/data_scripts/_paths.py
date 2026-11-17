"""Locations used by the data scripts: cached figure data in ../data, raw result files in ../results."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))


def data(name):
    return os.path.join(DATA, name)


def results(name):
    return os.path.join(RESULTS, name)
