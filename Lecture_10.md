# Lecture 10
## Perceptron
- ![Perceptron](./Images/Perceptron.png)
    - The activation function can be *generalized* to change the behavior of the perceptron
- ![Neural Networks](./Images/Neural_Networks.png)
## Architecture
- The architecture of a neural network is just the **network topology**, which is related to the number of units in the input layer, the number of hidden layers, the number of units in each hidden layer (if any), connections between layers, and the number of units in the output layer
    - This boils down to what maps the *input* to the *output* - so all the neurons and weights in between
## Activation Function
- The activation function controls the nature of the output of a layer (or output layer) of a neural network
- Activation functions bring *nonlinearity* into the hidden layers, which increases the complexity of the model
    - It should be noted, though, that feasible activation functions should be *differentiable*
- ![Activation Functions](./Images/Activation_Functions.png)
    - ReLU is most commonly used, as it has a *constant gradient* which avoids the issue of vanishing gradients
## Loss Functions
- The **loss function** of a neural network relates how good the predicted outputs are compared to the labels (target)
    - $L(w) = \frac{1}{n}\sum_i l(y^{(i)}, \hat{y}^{(i)})$, where $\hat{y}^{(i)}$ is the output of the neural network for $x_i$
- Common Loss Functions:
    - Squared Error: $l(y, \hat{y})=(y-\hat{y})^2$
    - Cross Entropy Loss: $l(y, \hat{y})= -y \log{\hat{y}} - (1 - y)\log{(1-\hat{y})}$
    - Hinge Loss: $l(y, \hat{y})= \max(0, 1 - y \hat{y})$
- With this loss, the empirical risk is minimized - this is done via *gradient descent*
    - For a multi-layered neural network, calculating the gradient involves making use of the *chain rule* to compute the gradient for the parameters of each layer - this is known as **backpropagation**