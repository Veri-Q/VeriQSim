from ast import Try
import datetime
import gc
import math
import os
import random
import time
import scipy
import numpy as np
import tensornetwork as tn
import signal
from contextlib import contextmanager
from qiskit import QuantumCircuit, Aer
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.quantum_info import Statevector
from angle_gen import faulty_gate_2
class TimeoutException(Exception): pass
from error_gen import *
from tn_construction import *
from verification_by_qiskit import *
import socket
# noise_gate = np.zeros((2, 2, 2, 2), dtype=complex)
# noise_gate[0][0][0][0] = 1
# noise_gate[0][1][0][1] = noise_gate[1][0][1][0] = 0.98224288
# noise_gate[1][1][1][1] = 0.99750312
# noise_gate[1][0][0][1] = 0.00249688
# noise_gate = np.array(noise_gate)

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def folder_test(path, output_file, error_num = 0):
    files = os.listdir(path)
    for f in files:
        try:
            cir_tn = QCTN(f, path)
            cir_tn.io_test(output_file, error_num, random_pos=True)
        except:
            raise
        gc.collect()

def folder_enum_test(path, output_file, error_num = 0, enum = 1):
    files = os.listdir(path)
    for f in files:
        try:
            cir_tn = QCTN(f, path)
            cir_tn.enumurate_io(output_file, error_num, random_pos=False, enum = enum)
        except:
            pass
        gc.collect()

def angle_test(path, filename, output_file):
    global theta, fault_crz
    lambda10 = 1.0866
    lambda20 = -0.0866
    tmax = 77.0
    g = math.sqrt(2)/40
    thetaf = math.pi/2
    thetai = math.atan(2*g/(0.31))

    def theta_noise(t, t0, lambda10, lambda20): #Fourier approximation of Slepian using 2 elements
        lambda1 = (thetaf-thetai)/t0*lambda10
        lambda2 = (thetaf-thetai)/t0*lambda20
        if( 0 <= t < t0):
            return thetai + ((lambda1+lambda2)*t - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*t/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*t/t0))
        elif (t0 <= t <= 2*t0):
            return (thetaf) - ((lambda1+lambda2)*(t-t0) - lambda1*t0/(2*math.pi)*math.sin(2*math.pi*(t-t0)/t0)-lambda2*t0/(4*math.pi)*math.sin(4*math.pi*(t-t0)/t0))   
    def faulty_gate_2(noise, tmax, index):
        def theta_noise1_value(x):
            return math.tan(theta_noise(x, tmax/2, lambda10 * (1 + noise), lambda20)/2)
        
        def theta_noise2_value(x):
            return math.tan(theta_noise(x, tmax/2, lambda10 + noise, lambda20 )/2)
        def theta_value(x):
            return math.tan(theta_noise(x, tmax/2, lambda10, lambda20)/2)
        
        sg2 = 2 * math.pi * g
        varphi1 = scipy.integrate.quad(theta_noise1_value, 0, tmax)[0]
        varphi2 = scipy.integrate.quad(theta_noise2_value, 0, tmax)[0]
        varphi3 = scipy.integrate.quad(theta_value, tmax * abs(noise/2), tmax * (1 - abs(noise)/2))[0]
        varphi = scipy.integrate.quad(theta_value,0,tmax)[0]
        if(index == 1): 
            return sg2 * (varphi1 - varphi)
        elif(index == 2):
            return sg2 * (varphi2 - varphi)
        elif(index == 3):
            return sg2 * (varphi3 - varphi)
    
    cir_tn = QCTN(filename, path)
    num_samples = 9
    X = np.linspace(-0.2, 0.2, num_samples)
    Y1 = np.zeros(num_samples)
    Y2 = np.zeros(num_samples)
    Y3 = np.zeros(num_samples)
    for index, noise in enumerate(X):
        tmax = 77.0
        Y1[index] = faulty_gate_2(noise, tmax,1)
        Y2[index] = faulty_gate_2(noise, tmax,2)
        Y3[index] = faulty_gate_2(noise, tmax,3)
    for Y in [Y1, Y2, Y3]:
        for index, noise in enumerate(X):
            theta = Y[index]
            cir_tn.set_angle(theta)
            with open(output_file, 'a') as f:
                time_now = datetime.datetime.now()
                print(time_now.strftime('%m.%d-%H:%M:%S'))
                f.write(time_now.strftime('%m.%d-%H:%M:%S')+"\n")
                f.write("Parameter: %f\t%f\t"%(X[index], theta))
            # cir_tn.single_cz_test(output_file)
            cir_tn.basis_test(output_file, all_crz=True)

def noise_number_test(path, file_name, output_file):
    for noise_number in range(80, 102, 2):
        try:
            cir_tn = QCTN(file_name, path)
            cir_tn.all_nodes = []
            cir_tn.enumurate_io(output_file, noise_number, enum = 1)
            # cir_tn.io_test(output_file, noise_number)
        except:
            raise
            pass
        gc.collect()

def noise_precision_test(path = DEFAULT_NOISE_PRECISION_PATH, filename = DEFAULT_NOISE_PRECISION_FILE, noise_number = 10, output_file = DEFAULT_NOISE_PRECISION_OUTPUT):
    cir_tn = QCTN(filename, path)
    cir_tn.fidelity_test(output_file, error_num=noise_number, random_pos=False)
    cir_tn.enumurate_fidility_test(output_file, noise_number, enum = 1, random_pos=False)

    # for enum in range(5, noise_number+1):
    #     try:
    #         cir_tn.all_nodes = []
    #         cir_tn.enumurate_fidility_test(output_file, noise_number, enum = enum, random_pos=False)
    #     except:
    #         raise
    #         pass
    #     gc.collect()

def decoherence_precision_test(path = DEFAULT_NOISE_PRECISION_PATH, filename = DEFAULT_NOISE_PRECISION_FILE, noise_number = 20, output_file = DEFAULT_NOISE_PRECISION_OUTPUT):
    cir_tn = QCTN(filename, path)
    cir_tn.fidelity_test(output_file, error_num=noise_number, random_pos=False)
    global T_ONE_TIME, T_TWO_TIME
    # for t1 in range(200, 510, 10):
    #     try:
    #         cir_tn.set_decoherence_time(t1, T_TWO_TIME)
    #         cir_tn.all_nodes = []
    #         # cir_tn.enumurate_fidility_test(output_file, noise_number, random_pos=False)
    #         cir_tn.fidelity_test(output_file, error_num=noise_number, random_pos=False)
    #     except:
    #         raise
    #         pass
    #     gc.collect()
    T_ONE_TIME = 200
    for t2 in range(20, 51):
        try:
            cir_tn.set_decoherence_time(T_ONE_TIME, t2)
            cir_tn.all_nodes = []
            cir_tn.fidelity_test(output_file, noise_number, random_pos=False)
        except:
            raise
            pass
        gc.collect()

def angle_value_test(path, filename, output_file):
    cir_tn = QCTN(filename, path)
    num_samples = 50
    X = np.linspace(0, 0.2, num_samples)
    for theta in X:
        gc.collect()
        cir_tn.set_angle(theta)
        with open(output_file, 'a') as f:
            # time_now = datetime.datetime.now()
            # print(time_now.strftime('%m.%d-%H:%M:%S'))
            # f.write(time_now.strftime('%m.%d-%H:%M:%S')+"\n")
            f.write("Parameter: %f\t"%theta)
        cir_tn.all_nodes = []
        cir_tn.fidelity_test(output_file, all_crz=True)

def decoherence_test(path, filename, output_file):
    cir_tn = QCTN(filename, path)
    num_samples = 49
    T2 = np.linspace(10, 20, num_samples)
    for t2 in T2:
        gc.collect()
        cir_tn.set_decoherence_time(200, t2)
        with open(output_file, 'a') as f:
            # time_now = datetime.datetime.now()
            # print(time_now.strftime('%m.%d-%H:%M:%S'))
            # f.write(time_now.strftime('%m.%d-%H:%M:%S')+"\n")
            f.write("%f\t"%t2)
        cir_tn.all_nodes = []
        cir_tn.fidelity_test(output_file, error_num=16, random_pos=True)

def function_test(func, cycles, path, filename = '', output_file = '', error_num = 0, enum = 1):
    for _ in range(cycles):
        gc.collect()
        time_now = datetime.datetime.now()
        print(time_now.strftime('%m.%d-%H:%M:%S'))
        with open(output_file, 'a') as f:
            f.write(time_now.strftime('%m.%d-%H:%M:%S')+"\n")
            f.write("%s\t%s\t%s\n"%(str(func), filename, output_file))
        if func == folder_enum_test:
            func(path, output_file, error_num, enum)
        if func == folder_test:
            func(path, output_file, error_num)
        if func == noise_number_test:
            func(path, filename, output_file)

class Experiment_Tests:
    server_name = socket.gethostname()
    def get_qaoa_folder_test(path = 'Benchmarks/QAOA2/', output_file = "TN_result_qaoa_folder_test_%s.txt"%server_name, error_num = 20):
        function_test(folder_test, 5, path, output_file=output_file, error_num = error_num)
    def get_qaoa_folder_enum_test(path = 'Benchmarks/QAOA2/', output_file = "TN_result_qaoa_folder_enum_test_%s.txt"%server_name, error_num = 20):
        function_test(folder_enum_test, 5, path, output_file=output_file, error_num = error_num)
    def get_vqe_folder_test(path = 'Benchmarks/HFVQE/', output_file = "TN_result_vqe_folder_test_%s.txt"%server_name, error_num = 20):
        function_test(folder_test, 5, path, output_file=output_file, error_num = error_num)
    def get_vqe_folder_enum_test(path = 'Benchmarks/HFVQE/', output_file = "TN_result_vqe_folder_enum_test_%s.txt"%server_name, error_num = 20):
        function_test(folder_enum_test, 5, path, output_file=output_file, error_num = error_num)
    def get_qaoa_noise_number_test(path = 'Benchmarks/QAOA2/', file = 'qaoa_100.qasm', output_file = "TN_result_qaoa_noise_number_test_%s.txt"%server_name):
        function_test(noise_number_test, 5, path, file, output_file=output_file) 
    

if __name__ == '__main__':
    # error_num = 2
    server_name = socket.gethostname()
    # error_pos = [i for i in range(error_num)]
    tn.set_default_backend("pytorch")

    folder_enum_test('Benchmarks/tmp/', 'TN_result_tmp_folder_enum_test_%s.txt'%server_name, error_num = NOISE_NUM, enum = 1)
    # set_fix_error(error_num)
    # function_test(folder_enum_test, 5, 'Benchmarks/QAOA/', output_file="TN_result_qaoa_%s.txt"%server_name, error_num = error_num)
    # function_test(folder_test, 5, 'Benchmarks/QAOA2/', output_file="TN_result_qaoa_folder_test_%s.txt"%server_name, error_num = error_num)
    # function_test(folder_test, 1, 'Benchmarks/Test/', output_file="TN_result_qaoa_%s.txt"%server_name, error_num = error_num)
    # function_test(folder_enum_test, 1, 'Benchmarks/QAOA2/', output_file="TN_result_qaoa_%s.txt"%server_name, error_num = error_num)
    # qiskit_simulate('Benchmarks/Test/', 'qmy.qasm', output_file="TN_result_qaoa_%s.txt"%server_name, error_num = error_num, error_position=[2, 3])
    
    # function_test(noise_number_test, 1, 'Benchmarks/QAOA2/', 'qaoa_100.qasm',output_file="TN_result_qaoa_%s.txt"%server_name, error_num = error_num)
    # function_test(folder_enum_test, 5, 'Benchmarks/inst_TN/', output_file="TN_result_inst_%s.txt"%server_name, error_num = error_num)
    # function_test(folder_test, 5, 'Benchmarks/inst_TN/', output_file="TN_result_inst_%s.txt"%server_name, error_num = error_num)
    # function_test(noise_number_test, 5, 'Benchmarks/inst_TN/', 'inst_6x6_20.qasm', output_file="TN_result_inst_%s.txt"%server_name, error_num = error_num)

    # noise_precision_test()
    # decoherence_precision_test()
    # Experiment_Tests.get_qaoa_noise_number_test()
    # Experiment_Tests.get_qaoa_folder_test()
    # Experiment_Tests.get_qaoa_folder_enum_test()
    # Experiment_Tests.get_vqe_folder_test()
    # Experiment_Tests.get_vqe_folder_enum_test()

    