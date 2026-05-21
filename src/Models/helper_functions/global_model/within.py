import numpy as np
import tensorflow as tf

class WithinHelper:
    def __init__(self, x_input):
        self.x_input = x_input.numpy()
        
    def calculate_P_matrix(self):
        
        """
        Function to calculate the P matrix for within transformation. as in 

        Returns
            * P_matrix: P matrix for within transformation.
        """
    
        # shapes, note that delta 1 and 2 have batch size 1 as first dimension
 
        
        
        Delta_1, Delta_2 = self.delta_fn()

        n = Delta_1.shape[0]
     
        Delta_N = Delta_1.T @ Delta_1 
        Delta_T = Delta_2.T @ Delta_2
        Delta_TN = Delta_2.T @ Delta_1
        Delta_N_inv = np.linalg.inv(Delta_N)
        
       
        # Note: Delta_TN.T is (N x T)
        D_bar = Delta_2 - Delta_1 @ Delta_N_inv @ Delta_TN.T

       
        Q = Delta_T - Delta_TN @ Delta_N_inv @ Delta_TN.T
        Q_inv = np.linalg.inv(Q)

        # Identity (n x n)
        I_n = np.eye(n, dtype=np.float16)
        
        #calculate p matrix components
        second_term= np.linalg.multi_dot([Delta_1, Delta_N_inv, Delta_1.T])
        third_term = np.linalg.multi_dot([D_bar, Q_inv, D_bar.T])
        

        P = I_n - second_term - third_term
        
        return P
    
    
    def delta_fn(self):
    
            x=self.x_input[0,:,:]

            N = x.shape[1]
            T = x.shape[0]
            
            #find where the NaNs are
            where_mat = np.isnan(x).T
           
            
            for t in range(T):
                idx = np.where(~where_mat[:, t])
                idx = np.reshape(idx, (-1,))

                D_t = np.eye(N)
                D_t = D_t[idx, :]
                

                if t == 0:
                    Delta_1 = D_t

                    Delta_2 = D_t @ np.ones((N, 1))

                else:
                    Delta_1 = np.vstack([Delta_1, D_t])

                    Delta_2 = np.hstack((Delta_2, np.zeros((np.shape(Delta_2)[0], 1))))

                    Delta_2_tmp = D_t @ np.ones((N, 1))
                    Delta_2_tmp = np.hstack((np.zeros((np.shape(Delta_2_tmp)[0], t)), Delta_2_tmp))

                    Delta_2 = np.vstack((Delta_2, Delta_2_tmp))

            return Delta_1, Delta_2
        
        
    @staticmethod
    @tf.function
    def within_twfe_balanced(v,T,N):
        """
        Two-way within transform for balanced panel, countries-fast stacking.
        v: shape (1, T*N, 1) or (T*N,) or (T*N,1)
        Returns same shape as input (flattened).
        """
        tf.print("Input shape:", v.shape)  # Debug print
        v = tf.convert_to_tensor(v, dtype=tf.float32)

        # Normalize shapes to (T*N,)
        if v.shape.rank == 3:
            v_flat = tf.reshape(v, [T * N])
            add_batch = True
        else:
            v_flat = tf.reshape(v, [T * N])
            add_batch = False

        tf.print("v_flat shape:", v_flat.shape)  # Debug print
        tf.print("T:", T, "N:", N)  # Debug print
        X = tf.reshape(v_flat, [T, N])  # countries-fast => (time, country), C-order

        time_mean    = tf.reduce_mean(X, axis=1, keepdims=True)  # (T,1)
        country_mean = tf.reduce_mean(X, axis=0, keepdims=True)  # (1,N)
        grand_mean   = tf.reduce_mean(X)

        X_dd = X - time_mean - country_mean + grand_mean

        out = tf.reshape(X_dd, [T * N])

        if add_batch:
            return tf.reshape(out, [1, T * N, 1])
        return out


