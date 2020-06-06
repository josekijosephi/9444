import numpy
import pandas
from csv import reader
from helper import scale, rename
from mathematical import sigmoid, loss
from forwardpass import forwardPass
from gradientdescent import gradientDescent
from test import test

# read from csv and store into vectors
db = pandas.read_csv('../data/iris.data.shuf.three', header=None)
db.drop(columns=[0,1], inplace=True)
db.columns = ['Length', 'Width', 'Class']


db.Class = db.Class.apply(rename)

inputs = numpy.array([1,db.loc[0][0], db.loc[0][1], db.loc[0][2]])

for i in range (1, db.shape[0]):
    inputs = numpy.vstack((inputs, numpy.array([1,db.loc[i][0], db.loc[i][1], db.loc[i][2]])))

df = pandas.DataFrame(inputs)
df.columns = [ 'Bias', 'Length', 'Width', 'Class' ]

# get the min and max of each column
len_max, width_max = df[['Length', 'Width']].max(axis=0)
len_min, width_min = df[['Length', 'Width']].min(axis=0)


df['Length'] = df['Length'].apply(scale, args=[len_max, len_min])
df['Width'] = df['Width'].apply(scale, args=[width_max, width_min])

# how we must init the 
weights = numpy.random.rand(3,3)

splice = 125
Xtrain = df[['Bias', 'Length', 'Width']].loc[:splice]
Xtrain = Xtrain.astype(float)
Ytrain = df['Class'].loc[:splice]
Xtest = df[['Bias', 'Length', 'Width']].loc[splice:]
Xtest = Xtest.astype(float)
Ytest = df['Class'].loc[splice:]

for i in range(len(Xtrain)):
    output, hot_one =forwardPass(Xtrain.loc[i], weights)
    gradientDescent(Xtrain.loc[i], weights, output, Ytrain[i])
    if i % 20 == 0:
        print(f"loss at iteration {i} is: {loss(output, df['Class'][i])}")

test(Xtest, Ytest, weights, splice)
