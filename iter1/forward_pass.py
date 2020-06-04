from numpy import argmax

def fp (Xin, weights):
    ''' Takes input to the transfer function
        Xin is the petal length and width,
        weights is the weight matrix'''
    output = []
    for i in range(3):
        c = weights[i][0] + weights[i][0] * Xin[1] + weights[i][2] * Xin[1]
        output.append(c)

    hot_one = []
    pos = argmax(output)

    for i in range(len(output)):
        if i == pos:
            hot_one.append(1)
        else:
            hot_one.append(0)
    #for entry in output:
    #    if entry > 0:
    #        hot_one.append(1)
    #    else:
    #        hot_one.append(0)
    
    return output, hot_one
