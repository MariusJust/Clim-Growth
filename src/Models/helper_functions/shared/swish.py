from tensorflow.keras.backend import sigmoid
def Swish(x, beta=1):
    """
    Swish activation function.

    ARGUMENTS
        * x:    input variable.
        * beta: hyperparameter of the Swish activation function.

    Returns
        * Swish activation function applied to x.
    """

    return x * sigmoid(beta * x)   
        