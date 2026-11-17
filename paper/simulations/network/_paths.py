"""Locations used by the network simulations: cached figure data in paper/data, raw results in paper/results,
scratch output in work/ (not part of the repository)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))   # paper/
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
WORK = os.path.join(HERE, 'work')


def data(name):
    return os.path.join(DATA, name)


def results(name):
    return os.path.join(RESULTS, name)


def work(name):
    return os.path.join(WORK, name)
