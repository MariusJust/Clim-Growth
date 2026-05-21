import tensorflow as tf
from .within import  WithinHelper

def individual_loss(mask, p_matrix=None, n_holdout=0, balanced=False):
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
        time_len = tf.shape(y_pred)[1]
        n_obs = y_true.shape[1]
        if n_holdout > 0:
            if n_obs == n_holdout:
                # HOLDOUT = last time_len
                mask_dyn = mask[:, -time_len:, :]

                if balanced:
                    # y_pred is (1, time_len, N) -> flatten countries-fast
                    y_pred = tf.reshape(y_pred, [1, time_len * tf.shape(y_pred)[2], 1])
                  
                    y_demeaned = WithinHelper.within_twfe_balanced(y_pred, time_len, tf.shape(y_pred)[2])
                    # Select observed (if balanced mask is all False, this is just v_dm)
                    y_pred_transf = tf.reshape(y_demeaned[~tf.reshape(mask_dyn, [1, time_len * tf.shape(y_pred)[2], 1])], (1, -1, 1))
                else:
                 
                    y_pred_transf = tf.reshape(y_pred[~mask_dyn], (1, -1, 1))
                    P_dyn = p_matrix[-n_holdout:, -n_holdout:]
                    y_pred_transf = tf.matmul(P_dyn, tf.cast(y_pred_transf, tf.float32))

            else:
                # TRAIN = first time_len
                mask_dyn = mask[:, :time_len, :]

                if balanced:
                    y_pred_dyn = y_pred[:, :time_len, :]
                    y_pred_reshaped = tf.reshape(y_pred_dyn, [1, time_len * tf.shape(y_pred)[2], 1])
                   
                    y_demeaned = WithinHelper.within_twfe_balanced(y_pred_reshaped, time_len, tf.shape(y_pred)[2])
                    y_pred_transf = tf.reshape(y_demeaned[~tf.reshape(mask_dyn, [1, time_len * tf.shape(y_pred)[2], 1])], (1, -1, 1))
                else:
                    y_pred_dyn = y_pred[:, :time_len, :]
                    y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
                    P_dyn = p_matrix[:n_obs, :n_obs]
                    y_pred_transf = tf.matmul(P_dyn, tf.cast(y_pred_transf, tf.float32))

            loss = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred_transf), axis=1)
            return loss

        else:
            time_len = tf.shape(y_true)[1]
            mask_dyn = mask[:, :time_len, :]
            y_pred_dyn = y_pred[:, :time_len, :]
            y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
            loss = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred_transf), axis=1)
            return loss    
    return loss