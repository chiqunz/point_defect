import numpy as np
import numpy.linalg as LA


def sha4(node_gaussian):
    N = np.zeros((4, 4))   
    for i in range(4):
        x = node_gaussian[i, 0]
        y = node_gaussian[i, 1]
        tmp = 0.25 * np.array(
            [(1 - x) * (1 - y),
             (1 + x) * (1 - y),
             (1 + x) * (1 + y),
             (1 - x) * (1 + y)])
        N[:, i] = tmp
    return N


def dsha4(node_gaussian):
    dN = np.zeros((4, 2, 4))
    for i in range(4):
        x = node_gaussian[i, 0]
        y = node_gaussian[i, 1]
        tmp = 0.25 * np.array(
             [[y - 1, 1 - y, 1 + y, -1 - y],
              [x - 1, -1 - x, 1 + x, 1 - x]]).T
        dN[:, :, i] = tmp
    return dN  # (4,2,4)


def Jacobi(node_gaussian, coords, incidence):
    coord_local = coords[incidence, :]  # (_, 4, 2)
    dN = dsha4(node_gaussian)  # (4,2,4)
    Jacobian = np.tensordot(coord_local, dN, axes=([1, 0]))  # (_, 2,2,4)
    detJ = [LA.det(j[:, :, 0]) for j in Jacobian]
    detJ = np.array(detJ)
    return Jacobian, detJ


def dshahat4(node_gaussian, coords, incidence):
    J, detJ = Jacobi(node_gaussian, coords, incidence)
    dN = dsha4(node_gaussian)  # (4,2,4)
    res = []
    for i in range(J.shape[0]):
        res_tmp = []
        for j in range(4):
            J_inv = LA.inv(J[i, :, :, j])  # (2,2)
            res_tmp.append(np.tensordot(dN[:, :, j], J_inv, axes=([1, 1])))  # (4,2)
        res.append(np.stack(res_tmp, axis=-1))
    return np.array(res), detJ
