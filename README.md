The tried and tested Iris
=========================

This will simply chronicle what is learnt about neural networks throughout COMP9444.

### Beginnings

To start with - everyones favourite - the iris data set is used to make a simple perceptron, with performance improved as new methods 
are learned and implemented.

## :help navigation

### simple_perceptron

This is the first iteration of the neural networks covered in the course. It uses the simple perceptron learning rule for its single
layer neural network.

It appears that there is not enough training data to successfully adjust the weights for the three classifications of Iris - however it can relatively effectively classify two classes of Iris.

### gradient_descent

Very similar base code with the adjustment of how the adjustment of the weight matrix occurs. This time going to use gradient descent 
to improve how the machine learns as well as changing the activation function from a step function to the sigmoid function.
