"""Fixture for the eigsh eval case: a 1-D Neumann (free-end) Laplacian, which is singular."""
import numpy as np
import scipy.sparse as sp

N = 200
L = sp.diags_array([-np.ones(N - 1), np.r_[1, 2 * np.ones(N - 2), 1], -np.ones(N - 1)],
                   offsets=[-1, 0, 1], format="csc")
