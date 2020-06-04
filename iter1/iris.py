import numpy
from numpy.random import rand as r
from forward_pass import fp
from get_csv import get_csv
from adjust import weight_adjust
from test import test

# change this if you want to use a pre-existing weight matrix
model = False
model_path = 'weights.npy'

Xinputs, Yin = get_csv('../data/iris.data.shuf.two')

# turn names into numbers
Yout = []
for flower in Yin:
    if flower == 'Iris-setosa':
        Yout.append([1,0,0])
    if flower == 'Iris-versicolor':
        Yout.append([0,1,0])
    if flower == 'Iris-virginica':
        Yout.append([0,0,1])

# seperate some test data
splice = 10

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

if model:
    with open(model_path, 'wb') as f:
        numpy.save(f,weights)

test(Xtest, Ytest, weights)
