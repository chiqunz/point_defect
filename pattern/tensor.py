import numpy as np


def alter_ini():
    ans = np.zeros((3, 3, 3))
    ans[0, 1, 2] = 1
    ans[2, 0, 1] = 1
    ans[1, 2, 0] = 1
    ans[2, 1, 0] = -1
    ans[0, 2, 1] = -1
    ans[1, 0, 2] = -1
    return ans


def delta_ini():
    ans = np.ones(3)
    ans = np.diag(ans)
    return ans
