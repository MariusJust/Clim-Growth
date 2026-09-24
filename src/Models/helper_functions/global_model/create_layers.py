from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras.layers import Dense, Dropout

def create_hidden_layers(self, input_first):
    ''' Create hidden layers based on the model's depth and node configuration. '''
    
    self.hidden_1 = create_hidden_layer(self, self.node[0])
    hidden_1 = self.hidden_1(input_first)
        
    # Handle depth and subsequent layers
    if self.Depth > 1:
        self.hidden_2 = create_hidden_layer(self, self.node[1])
        hidden_2 = self.hidden_2(hidden_1)
        if self.dropout != 0:
            hidden_2=create_Dropout(self, hidden_2)
        if self.Depth > 2:
            self.hidden_3 = create_hidden_layer(self, self.node[2])
            hidden_3 = self.hidden_3(hidden_2)
            if self.dropout != 0:
                hidden_3=create_Dropout(self, hidden_3)
            input_last =  hidden_3
        else:
            input_last =  hidden_2
    else:
        input_last =  hidden_1
    
    return input_last



def create_hidden_layer(self, node):
    """ Create a hidden layer with the specified input and layer number. """
    kernel_initializer = he_normal()
    bias_initializer = Zeros()
    hidden_layer = Dense(node, activation=self.activation, use_bias=True,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    
    
    return hidden_layer


def create_output_layer(self, input_tensor):
    """Create the output layer.

    ``instance.model.output_initializer``:
      ``'zeros'`` (default)  the historical behaviour of this repo. Note the
          consequence: the output layer is ``use_bias=False``, so a zero kernel
          makes the network output identically zero at initialisation AND makes
          dL/dh = dL/dy * W_outᵀ = 0, i.e. the hidden layers receive *exactly*
          no gradient on the first step. Adam then moves W_out by ~lr per step,
          so hidden-layer gradients start scaled by |W_out| ~ 1e-3 and it takes
          ~100 steps before they are O(0.1). Survivable with patience=200, but
          it changes the early trajectory.
      ``'he_normal'``  what Bennedsen, Hillebrand & Jensen (2023) use
          (Functions/Dynamic_NN_model.py: kernel_initializer_4 = he_normal()),
          and what this repo's own REGIONAL model already uses. Gradients reach
          the hidden layers immediately.

    Default stays 'zeros' so every run made before 2026-09-01 reproduces exactly;
    set the key explicitly to change it.
    """
    which = str(getattr(self, "output_initializer", "zeros")).lower()
    if which in ("he_normal", "he"):
        init = he_normal()
    elif which in ("zeros", "zero"):
        init = Zeros()
    else:
        raise ValueError(f"unknown output_initializer {which!r}; use 'zeros' or 'he_normal'")
    self.output_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=init)
    return self.output_layer(input_tensor)

def create_Dropout(self, layer):
    """ Create a dropout layer with a rate specified in the model. """
    dropout = Dropout(rate=self.dropout)
    dropout_layer = dropout(layer)
    return dropout_layer


