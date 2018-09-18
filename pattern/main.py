#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pattern problem """

import sys
import numpy as np

from pattern.models import model


CASE = sys.argv[1]
B_FIX = sys.argv[2]


def main():
    incidence = np.load('incidence.npy')
    incidence = incidence.astype('int')
    coords = np.load('coordinates.npy')
    m = model(CASE, incidence, coords, B_fix=B_FIX)
    m.build(filename_k='k.npy', filename_B='B.npy')
    m.optimize()
    np.save('k', m.k)
    np.save('B', m.B)

if __name__ == "__main__":
    main()
