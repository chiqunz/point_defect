'''build a model and store the parameters'''

import numpy as np
import numpy.linalg as LA

from pattern.constant import P1, P2, EPSILON, K2, STEP_SIZE
from pattern.tensor import alter_ini
from pattern.FEM import sha4, dshahat4
from pattern.eta import deta, eta


def initialize_k(filename=None):
    k = np.load(filename)
    return k


def initialize_B(filename=None):
    B = np.load(filename)
    return B

alter = alter_ini()


class model:
    '''The model'''
    def __init__(self, case, incidence, coords, B_fix=True):
        self.incidence = incidence
        self.coords = coords
        self.num_ele = incidence.shape[0]
        self.num_node = coords.shape[0]
        self.num_gaussian = self.num_ele * 4
        sqrt3 = np.sqrt(3)
        self.node_gaussian = np.array([
            [-sqrt3 / 3, -sqrt3 / 3],
            [sqrt3 / 3, -sqrt3 / 3],
            [sqrt3 / 3, sqrt3 / 3],
            [-sqrt3 / 3, sqrt3 / 3]
        ])
        self.case = int(case)
        self.Bfix = B_fix
        self.k = None
        self.B = None
        self.energy = None
        self.energy_test = None
        self.step = STEP_SIZE
        self.space = self.coords[1][0] - self.coords[0][0]
        self.bc = np.load('bc.npy')

    def build(self, filename_k=None, filename_B=None):
        if self.case == -1:
            self.k = initialize_k(filename_k)
            self.B = initialize_B(filename_B)
        elif self.case == 0:
            self.k = np.ones((self.num_node, 2))
            self.k[:, 0] = 0
            self.B = np.zeros((self.num_node, 2, 2))
        else:
            self.k = np.random.rand(self.num_node, 2)
            self.k /= LA.norm(self.k, axis=1).reshape(self.num_node, 1)
            self.B = np.zeros((self.num_node, 2, 2))

        self.N = sha4(self.node_gaussian)  # (4,4)
        self.dNhat, self.detJ = dshahat4(self.node_gaussian, self.coords, self.incidence)

    def update_k_B(self):
        '''update k for mesh nodes given element idx'''
        k_inc = np.zeros((self.num_node, 2))
        B_inc = np.zeros((self.num_node, 2, 2))
        update_k = []
        update_B = []
        self.energy = 0
        self.energy_test = [0, 0, 0, 0, 0]

        for ele_idx in range(self.num_ele):
            incidence_local = self.incidence[ele_idx]
            k_local = self.k[incidence_local]  # (4,2)
            B_local = self.B[incidence_local]  # (4,2,2)
            gradk_ele = np.tensordot(k_local, self.dNhat[ele_idx], axes=([0, 0]))  # (2,2,4)
            gradB_ele = np.tensordot(B_local, self.dNhat[ele_idx], axes=([0, 0]))  # (2,2,2,4)
            k_ele = k_local.T @ self.N  # (2,4)
            B_ele = np.tensordot(B_local, self.N, axes=([0, 0]))  # (2,2,4)

            # 2*P1*k/(1-|k|)
            part1 = 2 * P1 * (1 - 1 / (LA.norm(k_ele, axis=0) + 1e-5))  # (4,_)
            part1 = (part1 * k_ele)  # (2,4)
            part1 = np.tensordot(self.N, part1, axes=[1, 1])  # (4,2)
            self.energy += np.sum((LA.norm(k_ele, axis=0) - 1) ** 2) * P1 * self.detJ[ele_idx]
            self.energy_test[0] += (np.sum((LA.norm(k_ele, axis=0) - 1) ** 2) * P1 * self.detJ[ele_idx])

            # 2*P2*e_mij*e_mst*k_j^M*N^M_j*N^N_s
            part2_1 = np.zeros((3, 3, 4))  # gradk_padding, (3,3,4)
            part2_1[:2, :2, :] = gradk_ele  # (3,3,4)
            part2_1 = np.tensordot(alter[2, :, :], part2_1, axes=([0, 1], [1, 0])).reshape(1, 4)  # (1,4)
            part2_2 = np.zeros((4, 3, 4))  # dNhat_padding, (4,3,4)
            part2_2[:, :2, :] = self.dNhat[ele_idx]  # (4,3,4)
            part2_2 = np.tensordot(alter[2, :, :], part2_2, axes=([0, 1]))  # (3,4,4)
            part2 = np.tensordot(part2_2, part2_1, axes=([2, 1]))  # (3,4,1)
            part2 = 2 * P2 * part2[:2, :, 0].T  # (4,2)

            # add energy P2*|curlk|^2 (optional)
            self.energy += np.sum(part2_1 ** 2) * P2 * self.detJ[ele_idx]
            self.energy_test[1] += np.sum(part2_1 ** 2) * P2 * self.detJ[ele_idx]

            # add energy K2*|gradk - B|^2
            part3 = (gradk_ele - B_ele)  # (2,2,4)
            tmp = np.sum(part3 ** 2) * K2 * self.detJ[ele_idx]
            self.energy += tmp
            self.energy_test[2] += tmp

            # 2*K2*(k_t,n - B_tn)*N^N_n
            part3 = 2 * K2 * np.tensordot(self.dNhat[ele_idx], part3, axes=([1, 2], [1, 2]))  # (4,2)

            partk = part1 + part2 + part3
            update_k.append(partk * self.detJ[ele_idx])

            if not self.Bfix:
                # -2*K2*(gradk-B)_mn*N^N
                part1 = gradk_ele - B_ele  # (2,2,4)
                part1 = -2 * K2 * np.tensordot(self.N, part1, axes=([1, 2]))  # (4,2,2)

                # 2*epsilon*curlB_mt*e_tsn*N^N_s
                part2_1 = np.zeros((3, 3, 3, 4))  # gradB padding, (3,3,3,4)
                part2_1[:2, :2, :2, :] = gradB_ele  # (3,3,3,4)
                part2_1 = np.tensordot(part2_1, alter[2, :, :], axes=([1, 2], [1, 0]))  # (3,4)
                tmp = np.sum(part2_1 ** 2) * EPSILON * self.detJ[ele_idx]
                self.energy += tmp
                self.energy_test[3] += tmp
                part2_2 =  np.zeros((4, 3, 4))  # dNhat_padding, (4,3,4)
                part2_2[:,:2,:] = self.dNhat[ele_idx]  # (4,3,4)
                part2_2 = np.tensordot(alter[2, :, :], part2_2, axes=([0, 1]))  # (3,4,4)
                part2 = np.tensordot(part2_1, part2_2, axes=([1, 2]))  # (3,3,4)
                part2 = 2 * EPSILON * part2[:2, :2, :]  # (2,2,4)
                part2 = np.transpose(part2, [2, 0, 1])  # (4,2,2)

                part3 = [
                    deta(LA.norm(B_ele[:, :, i]), self.space) / (LA.norm(B_ele[:, :, i]) + 1e-5) * B_ele[:, :, i]
                    for i in np.arange(4)
                ]
                part3 = np.array(part3)  # (4,2,2)
                part3 = np.tensordot(self.N, part3, axes=([1, 0]))  # (4,2,2)

                tmp = np.sum([
                    eta(LA.norm(B_ele[:, :, i]), self.space) for i in np.arange(4)
                ]) * self.detJ[ele_idx]
                self.energy += tmp
                self.energy_test[4] += tmp

                partB = part1 + part2 + part3
                update_B.append(partB * self.detJ[ele_idx])

        for s in range(2):
            tmp = k_inc[:, [s]]
            np.add.at(tmp, self.incidence.flatten(), np.vstack(update_k)[:, [s]])
            k_inc[:, [s]] = tmp
            if not self.Bfix:
                for t in range(2):
                    tmp = B_inc[:, [s], [t]]
                    np.add.at(tmp, self.incidence.flatten(), np.vstack(update_B)[:, [s], [t]])
                    B_inc[:, [s], [t]] = tmp
        if not self.Bfix:
            return k_inc, B_inc
        return k_inc, 0

    def optimize(self):
        energy_old = 1e10
        energy_old_old = 1e10
        self.energy_rate = 1
        iter_idx = 0
        k_backup = np.copy(self.k)
        B_backup = np.copy(self.B)
        step_size_monitor = 0
        while self.energy_rate > 0.001:
            k_inc, B_inc = self.update_k_B()
            if self.energy > energy_old:
                self.k[:] = k_backup
                self.B[:] = B_backup
                step_size_monitor = 0
                energy_old = energy_old_old
                self.step *= 0.5
                step_size_monitor = 0
                print(f'shrink step size: {self.step}')
                continue
            k_backup[:] = self.k
            k_inc[self.bc] = 0
            self.k -= self.step * k_inc
            if not self.Bfix:
                B_backup[:] = self.B
                B_inc[self.bc] = 0
                self.B -= self.step * B_inc
            self.energy_rate = np.abs(energy_old - self.energy) / self.energy / self.step
            if iter_idx % 10 == 0:
                print([self.energy, self.energy_rate])
                print(self.energy_test)
            energy_old_old = energy_old
            energy_old = self.energy
            iter_idx += 1
            step_size_monitor += 1
            if step_size_monitor == 50:
                self.step *= 2
                step_size_monitor = 0
        return
