from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras.layers import Dense, Dropout

def create_hidden_layers(self, input_first):
    ''' Create hidden layers based on the model's depth and node configuration. '''

    self.hidden_1 = create_hidden_layer(self.node[0])
    hidden_1 = self.hidden_1(input_first)

    if self.Depth > 1:
        self.hidden_2 = create_hidden_layer(self.node[1])
        hidden_2 = self.hidden_2(hidden_1)
        if self.dropout != 0:
            hidden_2=create_Dropout(self, hidden_2)
        if self.Depth > 2:
            self.hidden_3 = create_hidden_layer(self.node[2])
            hidden_3 = self.hidden_3(hidden_2)
            if self.dropout != 0:
                hidden_3=create_Dropout(self, hidden_3)
            input_last =  hidden_3
        else:
            input_last =  hidden_2
    else:
        input_last =  hidden_1

    return input_last


def create_hidden_layer(node):
    kernel_initializer = he_normal()
    bias_initializer = Zeros()
    hidden_layer = Dense(node, activation='swish', use_bias=True,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    return hidden_layer


def create_output_layer(self, input_tensor):
    kernel_initializer = he_normal()
    self.output_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=kernel_initializer)
    return self.output_layer(input_tensor)

def create_Dropout(self, layer):
    dropout = Dropout(rate=self.dropout)
    dropout_layer = dropout(layer)
    return dropout_layer
