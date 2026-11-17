"""Locations used by the figure scripts.

The scripts can be started from any directory. Cached data are read from ../data,
intermediate GLE/LaTeX files go to build/ and the figure PDFs to output/.
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))
BUILD = os.path.join(HERE, 'build')
OUTPUT = os.path.join(HERE, 'output')
# GLE executable (https://glx.sourceforge.io); override with the environment variable GLE.
GLE = os.environ.get('GLE') or shutil.which('gle') or 'gle'


def enter_build():
    """Create build/ and output/ and make build/ the working directory."""
    os.makedirs(os.path.join(BUILD, 'gle'), exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)
    os.chdir(BUILD)


def data(name):
    return os.path.join(DATA, name)


def output(name):
    return os.path.join(OUTPUT, name)


def deliver(built, name):
    """Move a PDF produced in build/ to output/<name> and report it."""
    shutil.move(built, output(name))
    print('wrote', os.path.relpath(output(name), os.getcwd()))
