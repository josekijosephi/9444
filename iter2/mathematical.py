from numpy import exp

def sigmoid(x):
    return 1.0 / (1 + exp(-x)) 

def loss(Ycal, Yexp):
    summation = 0
    for i in range(len(Ycal)):
        summation += ( Ycal[i] - Yexp[i] ) ** 2

    return 1/2 * summation
