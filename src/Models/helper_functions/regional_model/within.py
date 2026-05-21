import numpy as np

class WithinHelper:
    def __init__(self, x_input):
        self.x_input = x_input

    def calculate_P_matrix(self):
        """
        Calculate the P matrix for within transformation for each region.

        Returns
            * P_list: list of P matrices, one per region.
        """
        P_list = []

        for i in enumerate(self.x_input):
            P_region = self.calculate_P_matrix_region(i[0])
            P_list.append(P_region)

        return P_list

    def calculate_P_matrix_region(self, i):

        Delta_1, Delta_2 = self.delta_fn(self.x_input[i].numpy()[0,:,:])
        n = Delta_1.shape[0]

        Delta_N = Delta_1.T @ Delta_1
        Delta_T = Delta_2.T @ Delta_2
        Delta_TN = Delta_2.T @ Delta_1
        Delta_N_inv = np.linalg.inv(Delta_N)

        D_bar = Delta_2 - Delta_1 @ Delta_N_inv @ Delta_TN.T

        Q = Delta_T - Delta_TN @ Delta_N_inv @ Delta_TN.T
        Q_inv = np.linalg.inv(Q)

        I_n = np.eye(n, dtype=np.float32)

        second_term= Delta_1 @ Delta_N_inv @ Delta_1.T
        third_term = D_bar @ Q_inv @ D_bar.T

        P = I_n - second_term + third_term

        return P

    def delta_fn(self, x):

            N = x.shape[1]
            T = x.shape[0]

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
