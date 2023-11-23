import numpy as np
import math
from config import *

def cRz(theta):
    mat = np.zeros((2, 2, 2, 2), dtype=complex)
    mat[0][0][0][0] = 1
    mat[0][1][0][1] = 1
    mat[1][0][1][0] = np.exp(-1.0j * theta /2)
    mat[1][1][1][1] = np.exp(1.0j * theta /2)
    return np.array(mat)

def gen_noise_gate(T1, Tphi, t):
    a = math.exp(-t/T1)
    b = math.exp(-t/Tphi)
    G1 = np.matrix([[1,0],[0,math.sqrt(a)]])
    G2 = np.matrix([[0,math.sqrt(1-a)],[0,0]])

    F1 = np.matrix([[math.sqrt(b),0],[0,math.sqrt(b)]])
    F2 = np.matrix([[math.sqrt(1-b),0],[0,0]])
    F3 = np.matrix([[0,0],[0,math.sqrt(1 - b)]])

    E1 = F1 * G1
    E2 = F2 * G1
    E3 = F3 * G1
    E4 = F1 * G2
    E5 = F2 * G2
    E6 = F3 * G2

    E = [E1, E2, E3, E4, E5, E6]

    res = np.zeros((4, 4), dtype=complex)
    for e in E:
        res += np.kron(e, e.H.T)
    return np.reshape(res, (2, 2, 2, 2))

def gen_depolorizing_gate(p):
    return np.array([[1-2*p/3   , 0         , 0         , 0],
                     [0         , 1-4*p/3   , 2*p/3     , 0],
                     [0         , 2*p/3     , 1-4*p/3   , 0],
                     [0         , 0         , 0         , 1-2*p/3]])

def mat_rep_svd(M):
    M = np.transpose(M, (0, 2, 1, 3))
    M = np.reshape(M, (4, 4))
    u, s, vh = np.linalg.svd(M)
    res = [[], s.copy(), []]
    for i in range(4):
        ui = np.reshape(u[: , i], (2, 2))
        vhi = np.reshape(vh[i, :], (2, 2))
        res[0].append(ui)
        res[2].append(vhi)
    return res

def mat_distance(M1, M2):
    M1 = np.reshape(M1, (4, 4))
    M2 = np.reshape(M2, (4, 4))
    return np.linalg.norm(M1 - M2, 2)

def distance_with_identity(M):
    identity = np.eye(4, dtype=complex)
    return mat_distance(M, identity)

def veri_svd(res):
    ans = np.zeros((4, 4), dtype='complex128')
    for i in range(4):
        ui = res[0][i]
        vhi = res[2][i]
        si =res[1][i]
        print(np.kron(ui, vhi) * si)
        ans += np.kron(ui, vhi) * si
    return ans


if __name__ == '__main__':
    # M = gen_noise_gate(T_ONE_TIME, T_TWO_TIME, DELTA_T)
    # p = distance_with_identity(M)
    # print(p)
    # bound = (1+8*p)**10-1-10*4*p*(1+4*p)**9
    # print(bound)
    
    M = gen_depolorizing_gate(0.0001)
    p = distance_with_identity(M)
    print(p)



