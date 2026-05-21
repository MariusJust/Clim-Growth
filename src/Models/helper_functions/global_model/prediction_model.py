from tensorflow.keras import Model
from .create_layers import create_Dropout
from tensorflow.keras.layers import Input, Add, Reshape

def pred_model(self):
    n_vars = len(self.input_vars)
    
    if self.dynamic_model:
        input_x_pred  = Input(shape=(1, None, self.n_vars+1), name='X_in')
    else:
        input_x_pred  = Input(shape=(1, None, self.n_vars), name='X_in')
    


    hidden_1_pred = self.hidden_1(input_x_pred)
    if self.dropout !=0:
        hidden_1_pred = create_Dropout(self, hidden_1_pred)
    

    if self.Depth > 1:
        hidden_2_pred = self.hidden_2(hidden_1_pred)
        if self.dropout !=0:
            hidden_2_pred=create_Dropout(self, hidden_2_pred)

        if self.Depth > 2:
            hidden_3_pred = self.hidden_3(hidden_2_pred)
            if self.dropout !=0:
                hidden_3_pred=create_Dropout(self, hidden_3_pred)
            input_last_pred = hidden_3_pred
        else:
            input_last_pred = hidden_2_pred

    else:
        input_last_pred = hidden_1_pred

    output_pred = self.output_layer(input_last_pred)

    #if we are utilizing within transformation, we do not include fixed effects


    if self.holdout==0:
        country_FE= Input(shape=(1, self.N['global'], 1), name='country_FE_in')
        time_FE= Input(shape=(1, 1, 1), name='time_FE_in')
        
        if self.dynamic_model:
            fe_sum = Add(name='fe_sum')([output_pred,
                                    country_FE])
            final = Reshape((self.N['global'], 1), name='final_pred')(fe_sum)
            return Model(inputs=[input_x_pred, country_FE], outputs=final,
                        name='prediction_with_FE')
        else:
            # Adding fixed effects
            fe_sum = Add(name='fe_sum')([output_pred,
                                        country_FE,
                                        time_FE])
            final = Reshape((self.N['global'], 1), name='final_pred')(fe_sum)

            return Model(inputs=[input_x_pred, country_FE, time_FE], outputs=final,
                        name='prediction_with_FE')
    else:
        final = Reshape((self.N['global'], 1), name='final_pred')(output_pred)
        return Model(inputs=[input_x_pred], outputs=final,
                        name='prediction_within_transform')



