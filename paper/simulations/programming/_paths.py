"""Locations used by the programming simulations (figure 3, ladders, sigma table).

DATA: cached figure data (paper/data); RESULTS: shipped intermediate results (paper/results/programming);
WORK: scratch and per-N checkpoints (paper/simulations/programming/work, created on demand)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results', 'programming')
WORK = os.path.join(HERE, 'work')


def data(name):
    return os.path.join(DATA, name)


def results(name):
    return os.path.join(RESULTS, name)


def work(name):
    os.makedirs(WORK, exist_ok=True)
    return os.path.join(WORK, name)
