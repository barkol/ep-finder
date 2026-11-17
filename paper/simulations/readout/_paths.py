"""Locations used by the read-out simulations: intermediate results in ../../results/readout (shipped, so the aggregate
scripts rebuild the outputs in seconds; the sweeps skip every configuration already present there), final outputs in
../../data and ../../results, inputs in inputs/. Set READOUT_WORK to use another directory for intermediates."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, '..', '..'))
WORK = os.environ.get('READOUT_WORK') or os.path.join(PAPER, 'results', 'readout')


def work(name):
    return os.path.join(WORK, name)


def data(name):
    return os.path.join(PAPER, 'data', name)


def results(name):
    return os.path.join(PAPER, 'results', name)


def inputs(name):
    return os.path.join(HERE, 'inputs', name)
