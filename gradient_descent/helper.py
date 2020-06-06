import numpy

def scale(x, Max, Min):
    return ( x - Min ) / ( Max - Min )

def rename(flower):
    return numpy.array([(flower=='Iris-setosa'), (flower=='Iris-versicolor'),(flower=='Iris-virginica')])
