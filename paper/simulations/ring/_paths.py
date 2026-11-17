"""Locations used by the ring simulations: final outputs in ../../data and ../../results, checkpoints in work/."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
WORK = os.path.join(HERE, 'work')


def data(name):
    return os.path.join(DATA, name)


def results(name):
    return os.path.join(RESULTS, name)


def work(name):
    os.makedirs(WORK, exist_ok=True)
    return os.path.join(WORK, name)
