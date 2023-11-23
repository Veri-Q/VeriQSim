import random

T_ONE_TIME = 200
T_TWO_TIME = 30
DELTA_T = 0.5
THETA = 0.5
NOISE_NUM = 2
# NOISE_POS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ,110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
NOISE_POS = [10, 20]


DEFAULT_NOISE_PRECISION_PATH = 'Benchmarks/QAOA2/'
DEFAULT_NOISE_PRECISION_FILE = 'qaoa_15.qasm'
DEFAULT_NOISE_PRECISION_OUTPUT = 'qaoa_15_noise_precision.txt'

DEFAULT_DECOHERENCE_PRECISION_PATH = 'Benchmarks/QAOA2/'
DEFAULT_DECOHERENCE_PRECISION_FILE = 'qaoa_64.qasm'
DEFAULT_DECOHERENCE_PRECISION_OUTPUT = 'qaoa_64_decoherence_precision.txt'

def set_fix_error(error_num):
    global NOISE_NUM, NOISE_POS
    NOISE_NUM = error_num
    l = [i for i in range(100)]
    NOISE_POS = random.sample(l, error_num)
    NOISE_POS.sort()