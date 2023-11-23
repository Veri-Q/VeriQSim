import numpy as np

def calculate_precision(p, n):
    return (1+8*p)**n-1-n*4*p*(1+4*p)**(n-1)

def sample_number(t):
    return np.log(100)/(2*t**2)

if __name__ == '__main__':
    # for p in [0.01, 0.001, 0.0001, 0.00001]:
    #     for n in range(10 , 41):
    #         delta = calculate_precision(p, n)
    #         # print("%f\t%f\t%f\t%f"%(p, n, delta, sample_number(delta)))
    #         print("(%d, %f)"%(n, sample_number(delta)))
    for n in range(10 , 41):
        print("(%d, %d)"%(n, 6*n+2))