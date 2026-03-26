
import tensorflow as tf 
from tensorflow.keras.layers import Layer
import numpy as np

# %% Creating vectorization layer
class Vectorize(Layer):
    """
    Layer that vectorizes the second dimension of inputs.
    """

    def __init__(self, N, variable, time_periods=None, **kwargs):
        super(Vectorize, self).__init__(**kwargs)
        self.N = N
        self.time_periods = time_periods
        
        self.dim1 = None
        self.variable=variable

    def call(self, x):
        mask = tf.math.is_nan(x)
            
        if self.variable=='time':
            time_periods=tf.reshape(self.time_periods, (1, -1, 1)) - (self.time_periods[0] - 1)
            time_mat = tf.repeat(time_periods, repeats=self.N['global'], axis=2)
            var_clean = tf.reshape(tf.cast(time_mat[~mask], dtype=np.float32), (1, -1, 1))

        else:
           
            # Apply the mask to both slices.
            var_clean = tf.reshape(x[~mask], (1, -1, 1))

            # Store the shape of the first variable for further processing
            self.dim1 = tf.shape(var_clean)[1]

        return var_clean
    
    
    def compute_output_shape(self):
        return (1, self.dim1, 1)
    
    
    
   