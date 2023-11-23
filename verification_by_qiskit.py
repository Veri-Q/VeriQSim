from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit_aer.noise import depolarizing_error
import qiskit.quantum_info as qi
import math, random
import numpy as np

from tn_construction import file_to_cir

def gen_noise_karus(T1, Tphi, t):
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
    return E

def apply_noise(circ, error_num, error_position, noise_op):
    cnt = 0
    pos = 0
    noise_circ = QuantumCircuit(circ.num_qubits)
    for gate in circ.data:
        operating_qubits = [x.index for x in gate[1]]
        noise_circ.append(gate[0], operating_qubits)
        if(cnt < error_num and pos == error_position[cnt]):
            error_qubit = operating_qubits[0]
            errors = [error_qubit]
            noise_circ.append(noise_op, errors)
            cnt += 1
        pos += 1
    return noise_circ

def qiskit_simulate(path, file_name, output_file ,error_num, error_position):
    simulator = Aer.get_backend('aer_simulator_density_matrix')
    circ = file_to_cir(file_name, path)
    noise_op = qi.Kraus(gen_noise_karus(200, 30, 0.5))
    noise_circ = apply_noise(circ, error_num, error_position, noise_op)
    noise_circ.save_state()
    result = simulator.run(noise_circ).result()
    print(result.data(0))

if __name__ == '__main__':
    num_qubits = 1
    rho = qi.DensityMatrix([1, 0])
    circ = QuantumCircuit(num_qubits)
    circ.set_density_matrix(rho)
    noise_op = qi.Kraus(gen_noise_karus(200, 50, 0.5))
    circ.x(0)
    circ.append(noise_op, [0])
    circ.save_state()

    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)

    # Run and get saved data
    result = simulator.run(circ).result()
    print(result.data(0))