from tensorflow.keras import Model
from tensorflow.keras.layers import Input


def Visual_model(self):
    if self.dynamic_model:
        input_x_pred  = Input(shape=(1, None, self.n_vars+1), name='X_in')
    else:
         input_x_pred  = Input(shape=(1, None, self.n_vars), name='X_in')
    
    hidden_1_pred = self.hidden_1(input_x_pred)
  
    if self.Depth > 1:
        hidden_2_pred = self.hidden_2(hidden_1_pred)
       
        if self.Depth > 2:
            hidden_3_pred = self.hidden_3(hidden_2_pred)
            
            input_last_pred = hidden_3_pred

        else:
            input_last_pred = hidden_2_pred

    else:
        input_last_pred = hidden_1_pred

    output_pred = self.output_layer(input_last_pred)


    model_visual = Model(
        inputs=[input_x_pred],
        outputs=output_pred,
        name='visual_model'
    )
    return model_visual

