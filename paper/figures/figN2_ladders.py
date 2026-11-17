#!/usr/bin/env python3
"""Figure N2 of the technical notes: ladders per cycle type. -> output/figN2_ladders.pdf"""
from paths import enter_build
enter_build()
import mpl_figures
mpl_figures.figN2()
print('wrote output/figN2_ladders.pdf')
