import numpy
from numpy.random import rand as r
from forward_pass import fp
from get_csv import get_csv
from adjust import weight_adjust, loss, column, scale
from test import test

# change this if you want to use a pre-existing weight matrix
model = False
model_path = 'weights.npy'

Xinputs, Yin = get_csv('../data/iris.data.shuf.three')

# turn names into numbers
Yout = []
for flower in Yin:
    if flower == 'Iris-setosa':
        Yout.append([1,0,0])
    if flower == 'Iris-versicolor':
        Yout.append([0,1,0])
    if flower == 'Iris-virginica':
        Yout.append([0,0,1])

# normalize size of data
len_max = max(column(Xinputs,0))
len_min = min(column(Xinputs,1))
width_max = max(column(Xinputs,0))
width_min = min(column(Xinputs,1))
for i in range(len(Xinputs)):
    Xinputs[i][0] = scale(Xinputs[i][0], len_max, len_min)
    Xinputs[i][1] = scale(Xinputs[i][1], width_max, width_min)

# seperate some test data
splice = 25

Xtrain = Xinputs[:-splice]
Ytrain = Yout[:-splice]

Xtest = Xinputs[-splice:]
Ytest = Yout[-splice:]


if not model:
    weights = r(3,3)
else:
    with open(model_path, 'rb') as f:
        weights = numpy.load(f)

# training loop time
for i in range(len(Xtrain)):
    output, hot_one = fp(Xtrain[i], weights)
    weight_adjust(Xtrain[i], hot_one, Ytrain[i], weights)
    if i % 20 == 0:
        calc_loss = loss(output, Ytrain[i])
        print(calc_loss)

if model:
    with open(model_path, 'wb') as f:
        numpy.save(f,weights)

test(Xtest, Ytest, weights)
