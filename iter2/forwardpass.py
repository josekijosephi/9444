from mathematical import sigmoid
from numpy import matmul, argmax, array

def forwardPass (Xin, weights):
    output = sigmoid(matmul(Xin, weights))

    hot_one = []
    aMax = argmax(output)
    hot_one = [ (aMax == 0),  (aMax == 1), (aMax == 2) ]

    return output, array(hot_one)
