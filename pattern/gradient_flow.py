import numpy as np
import numpy.linalg as LA

from script.FEM import sha4, dshahat4
from script.tensor import alter_ini, delta_ini
from script.algebra import contraction

def update_k_B(model):

    alter = alter_ini()
    delta = delta_ini()
    '''update k for mesh nodes given element idx'''
    k_inc = np.zeros((model.num_node, 2))
    B_inc = np.zeros((model.num_node, 2, 2))
    update_k = []
    update_B = []
    energy = 0
    for ele_idx in range(model.num_ele):
        incidence_local = model.incidence[ele_idx]
        k_local = model.k[incidence_local]
        B_local = model.B[incidence_local]
        partk = np.zeros((4, 2))
        partB = np.zeros((4, 2, 2))
        for int_idx in range(4):
            N = sha4(model.node_gaussian[int_idx, 0], model.node_gaussian[int_idx, 1])
            dNhat, detJ = dshahat4(model.node_gaussian[int_idx, 0],
                                   model.node_gaussian[int_idx, 1], ele_idx,
                                   model.incidence, model.coords)
            dNhat = dNhat.T
            gradk_local = (k_local.T @ dNhat).reshape(2, 2)
            gradB_local = (np.transpose(B_local, (1, 2, 0)) @ dNhat).reshape(2, 2, 2)

            part1 = 2 * model.P1 * (1 - 1 / (LA.norm(k_local.T @ N) + 1e-5))
            part1 = (part1 * N).reshape(4, 1)
            part1 = part1 * k_local
            energy += (LA.norm(k_local.T @ N) - 1) **2 * model.P1

            part2_tmp = np.zeros((3, 3))
            part2_tmp[:2, :2] = gradk_local
            part2 = 2 * model.P2 * alter[2, :, :] * part2_tmp.T
            part2 = np.sum(part2)
            energy += part2 ** 2 / 4 / model.P2
            part2_tmp = np.zeros((4, 3))
            part2_tmp[:, :2] = dNhat
            part2 = part2_tmp @ alter[:, :, 2] * part2
            part2 = part2[:, :2]

            part3 = 2 * model.K2 * (gradk_local -
                                    (np.transpose(B_local, (1, 2, 0)) @ N).reshape(2, 2))
            energy += LA.norm(part3) **2 / 4 / model.K2
            part3 = (part3 @ dNhat.T).T

            partk += part1 + part2 + part3

            part1 = -2 * model.K2 * (gradk_local -
                                     (np.transpose(B_local, (1, 2, 0)) @ N).reshape(1, 2, 2))
            part1 = part1.repeat(4, 0)*(N.reshape(4, 1, 1))

            part2_tmp = np.zeros((3, 3, 3))
            part2_tmp[:2, :2, :2] = gradB_local
            part2 = 2 * model.epsilon * contraction(part2_tmp, alter[2, :, :], [1, 2], [1, 0])
            part2 = part2[:2]
            energy += LA.norm(part2) ** 2 / 4 / model.epsilon
            part2_tmp = np.zeros((4, 3))
            part2_tmp[:, :2] = dNhat
            part2_tmp = (part2_tmp @ alter[:, :, 2])[:, :2]
            part2 = np.outer(part2_tmp, part2)
            part2 = part2.reshape(4, 2, 2)
            part2 = np.transpose(part2, (0, 2, 1))

            partB += part1 + part2

        update_k.append(partk * detJ)
        update_B.append(partB * detJ)

    for s in range(2):
        tmp = k_inc[:, [s]]
        np.add.at(tmp, model.incidence.flatten(), np.vstack(update_k)[:, [s]])
        k_inc[:, [s]] = tmp
        for t in range(2):
            tmp = B_inc[:, [s], [t]]
            np.add.at(tmp, model.incidence.flatten(), np.vstack(update_B)[:, [s], [t]])
            B_inc[:, [s], [t]] = tmp

    return k_inc, B_inc, energy
