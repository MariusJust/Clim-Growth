import tensorflow as tf

def individual_loss(mask, p_matrix=None, n_holdout=0, name=None):
    """
    Loss function (in two layers so that it can be interpreted by tensorflow).

    ARGUMENTS
        * mask: mask used to identify missing observations.

    Returns
        * loss: loss function evaluated in y_true and y_pred.
        
    """
    def loss(y_true, y_pred):
        """
        ARGUMENTS
            * y_true: observed targets.
            * y_pred: predicted targets.

        RETURNS
            * loss function evaluated in y_true and y_pred.
        """

        if p_matrix is not None:
            time_len=tf.shape(y_pred)[1]
            n_obs = y_true.shape[1]
        
            if n_obs == n_holdout:
                
                # slice mask to match y_true time-length
                mask_dyn = mask[:, -time_len:, :]                  
            
                # y_true_transf = tf.reshape(y_true[~mask_dyn], (1, -1, 1))
                y_pred_transf = tf.reshape(y_pred[~mask_dyn], (1, -1, 1))
                
                #count number of non-missing observations from y_true_transf

                P_dyn = p_matrix[-n_holdout:, -n_holdout:]
            else:
                
                mask_dyn = mask[:, :time_len, :]    
                                
                y_pred_dyn = y_pred[:, :time_len, :]
                
            
                y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
                
                P_dyn = p_matrix[:n_obs, :n_obs]

        
            y_pred_transf = tf.matmul(P_dyn, tf.cast(y_pred_transf, dtype=tf.float32))
            loss=tf.reduce_mean(tf.math.squared_difference(y_true, y_pred_transf), axis=1)
            return loss 
        else:
            time_len = tf.shape(y_true)[1]                    # e.g. 58 during training
            # slice mask to match y_true time-length
            
            mask_dyn = mask[:, :time_len, :]                  
               
            y_pred_dyn = y_pred[:, :time_len, :]
            y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
            y_true=tf.reshape(y_true[~mask_dyn], (1, -1, 1))
            
      
            loss=tf.reduce_mean(tf.math.squared_difference(y_true, y_pred_transf), axis=1)
            return loss
        
    if name is not None:
        loss.__name__ = name  #           

    return loss




